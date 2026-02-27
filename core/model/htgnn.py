import torch
import torch.nn as nn
from torch_geometric.nn import HGTConv, Linear


# ============================================================
# TEMPORAL ENCODER
# ============================================================
class TimeEncoder(nn.Module):
    # 
    #     Learnable Bochner time embedding (Xu et al., 2020).
    #     Maps scalar timestamps to d-dimensional representations.
    #     

    def __init__(self, d_model):
        super().__init__()
        self.d_model = d_model
        # Learnable frequency and phase
        self.w = nn.Linear(1, d_model)
        self.reset_parameters()

    def reset_parameters(self):
        # Initialize with diverse frequencies
        nn.init.xavier_uniform_(self.w.weight)
        nn.init.zeros_(self.w.bias)

    def forward(self, t):
        # 
        #         Args:
        #             t: Timestamps tensor, shape [N] or [N, 1]
        #         Returns:
        #             Time embeddings, shape [N, d_model]
        #         
        if t.dim() == 1:
            t = t.unsqueeze(-1)
        t = t.float()
        # Normalize timestamps to [0, 1] range
        t_min, t_max = t.min(), t.max()
        if t_max - t_min > 0:
            t = (t - t_min) / (t_max - t_min)
        output = torch.cos(self.w(t))
        return output


# ============================================================
# NODE ENCODERS
# ============================================================
class PayerEncoder(nn.Module):
    # Encode payer node features: issuer (emb) + country (emb) → linear.

    def __init__(self, n_issuers, n_countries, d_model, emb_dim_issuer=16,
                 emb_dim_country=8):
        super().__init__()
        self.issuer_emb = nn.Embedding(n_issuers, emb_dim_issuer)
        self.country_emb = nn.Embedding(n_countries, emb_dim_country)
        self.linear = nn.Linear(emb_dim_issuer + emb_dim_country, d_model)

    def forward(self, x):
        # x: [N_payer, 2] — columns: [issuer_idx, country_idx]
        issuer = self.issuer_emb(x[:, 0])
        country = self.country_emb(x[:, 1])
        return self.linear(torch.cat([issuer, country], dim=-1))


class MerchantEncoder(nn.Module):
    # Encode merchant node features: mcc (emb) + location (emb) + qris_type.

    def __init__(self, n_mcc, n_location, n_qris, d_model,
                 emb_dim_mcc=16, emb_dim_location=8, emb_dim_qris=4):
        super().__init__()
        self.mcc_emb = nn.Embedding(n_mcc, emb_dim_mcc)
        self.location_emb = nn.Embedding(n_location, emb_dim_location)
        self.qris_emb = nn.Embedding(n_qris, emb_dim_qris)
        self.linear = nn.Linear(emb_dim_mcc + emb_dim_location + emb_dim_qris,
                                d_model)

    def forward(self, x):
        # x: [N_merchant, 3] — columns: [mcc_idx, location_idx, qris_type_idx]
        mcc = self.mcc_emb(x[:, 0])
        loc = self.location_emb(x[:, 1])
        qris = self.qris_emb(x[:, 2])
        return self.linear(torch.cat([mcc, loc, qris], dim=-1))


# ============================================================
# EDGE CLASSIFIER
# ============================================================
class EdgeClassifier(nn.Module):
    # Simplified 2-layer MLP: src_emb || edge_feat || dst_emb → fraud score.

    def __init__(self, d_model, edge_feat_dim, hidden_dim=128, dropout=0.2):
        super().__init__()
        input_dim = d_model * 2 + edge_feat_dim
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, src_emb, edge_attr, dst_emb):
        # 
        #         Args:
        #             src_emb: [E, d_model] — payer embeddings for each edge
        #             edge_attr: [E, edge_feat_dim] — edge features
        #             dst_emb: [E, d_model] — merchant embeddings for each edge
        #         Returns:
        #             logits: [E] — raw scores (before sigmoid)
        #         
        h = torch.cat([src_emb, edge_attr, dst_emb], dim=-1)
        return self.mlp(h).squeeze(-1)


# ============================================================
# EDGE ANOMALY HEAD (per-edge, no GNN)
# ============================================================
class EdgeAnomalyHead(nn.Module):
    # Detects per-edge anomalies directly from edge features.
    #     
    #     This head bypasses the GNN entirely and scores edges based on their
    #     raw features (amount_zscore, velocity, hour, etc.). This catches
    #     fraud types that don't have structural signatures:
    #       - amount_anomaly: extreme amounts at off-hours
    #       - geo_anomaly: impossible travel (distant locations in short time)
    #     

    def __init__(self, edge_feat_dim, hidden_dim=64, dropout=0.2):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(edge_feat_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, edge_attr):
        # Args: edge_attr [E, edge_feat_dim]. Returns: logits [E].
        return self.mlp(edge_attr).squeeze(-1)


# ============================================================
# HTGNN MODEL
# ============================================================
class AegisHTGNN(nn.Module):
    # 
    #     Heterogeneous Temporal Graph Neural Network for fraud detection.
    # 
    #     Architecture:
    #         1. Node encoding: Embedding layers for categorical features
    #         2. Time encoding: Bochner temporal embedding
    #         3. GNN: 3 HGTConv layers with residual + LayerNorm
    #         4. Edge classifier: MLP on (src || edge_feat || dst)
    #     

    def __init__(self, meta, cat_sizes, edge_feat_dim=12,
                 d_model=64, n_heads=4, n_layers=3, dropout=0.3):
        # 
        #         Args:
        #             meta: Tuple (node_types, edge_types) from HeteroData
        #             cat_sizes: Dict with category sizes for embeddings
        #             edge_feat_dim: Number of edge features (12)
        #             d_model: Hidden dimension
        #             n_heads: Attention heads for HGTConv
        #             n_layers: Number of HGTConv layers
        #             dropout: Dropout rate
        #         
        super().__init__()

        self.d_model = d_model
        self.n_layers = n_layers

        # Node encoders
        self.payer_encoder = PayerEncoder(
            n_issuers=cat_sizes['source_issuer'],
            n_countries=cat_sizes['source_country'],
            d_model=d_model,
        )
        self.merchant_encoder = MerchantEncoder(
            n_mcc=cat_sizes['merchant_mcc'],
            n_location=cat_sizes['merchant_location'],
            n_qris=cat_sizes['qris_type'],
            d_model=d_model,
        )

        # Temporal encoder
        self.time_encoder = TimeEncoder(d_model)

        # Edge feature projection (for injecting into GNN)
        self.edge_proj = nn.Linear(edge_feat_dim, d_model)
        # Separate projections for source and destination edge injection
        self.edge_to_src = nn.Linear(d_model, d_model)
        self.edge_to_dst = nn.Linear(d_model, d_model)

        # HGTConv layers with residual
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        for _ in range(n_layers):
            conv = HGTConv(
                in_channels=d_model,
                out_channels=d_model,
                metadata=meta,
                heads=n_heads,
            )
            self.convs.append(conv)
            # Per-type layer norms
            norm_dict = nn.ModuleDict({
                nt: nn.LayerNorm(d_model) for nt in meta[0]  # node_types
            })
            self.norms.append(norm_dict)

        self.dropout = nn.Dropout(dropout)

        # Edge classifier (GNN-based: uses node embeddings + edge features)
        self.classifier = EdgeClassifier(
            d_model=d_model,
            edge_feat_dim=edge_feat_dim,
            hidden_dim=d_model * 2,
            dropout=dropout,
        )

        # Edge anomaly head (edge-only: bypasses GNN for per-edge anomalies)
        self.anomaly_head = EdgeAnomalyHead(
            edge_feat_dim=edge_feat_dim,
            hidden_dim=d_model,
            dropout=dropout,
        )

        # Learnable gate to blend GNN score vs edge-only score
        # Initialized to 0 → sigmoid(0) = 0.5 → equal blend
        self.gate = nn.Parameter(torch.zeros(1))

    def forward(self, x_dict, edge_index_dict, edge_attr_dict=None,
                timestamps=None):
        # 
        #         Args:
        #             x_dict: Dict of node features {node_type: tensor}
        #             edge_index_dict: Dict of edge indices {edge_type: tensor}
        #             edge_attr_dict: Dict of edge attributes (optional for HGTConv, 
        #                            used by classifier)
        #             timestamps: Edge timestamps for temporal encoding (optional)
        #         Returns:
        #             logits: [E] raw prediction scores for each edge
        #         
        edge_type = ('payer', 'TRANSACTS', 'merchant')

        # Step 1: Encode node features
        h_dict = {
            'payer': self.payer_encoder(x_dict['payer']),
            'merchant': self.merchant_encoder(x_dict['merchant']),
        }

        # Step 2: Add temporal signal to nodes (via mean of connected edge times)
        if timestamps is not None:
            edge_idx = edge_index_dict[edge_type]
            time_emb = self.time_encoder(timestamps)

            # Aggregate temporal signal to payer nodes
            payer_time = torch.zeros(h_dict['payer'].shape[0], self.d_model,
                                     device=h_dict['payer'].device)
            payer_count = torch.zeros(h_dict['payer'].shape[0], 1,
                                      device=h_dict['payer'].device)
            payer_time.scatter_add_(0, edge_idx[0].unsqueeze(1).expand_as(time_emb),
                                    time_emb)
            payer_count.scatter_add_(0, edge_idx[0].unsqueeze(1)[:, :1],
                                     torch.ones(edge_idx.shape[1], 1,
                                                device=h_dict['payer'].device))
            payer_count = payer_count.clamp(min=1)
            h_dict['payer'] = h_dict['payer'] + payer_time / payer_count

            # Aggregate temporal signal to merchant nodes
            merchant_time = torch.zeros(h_dict['merchant'].shape[0], self.d_model,
                                        device=h_dict['merchant'].device)
            merchant_count = torch.zeros(h_dict['merchant'].shape[0], 1,
                                          device=h_dict['merchant'].device)
            merchant_time.scatter_add_(0,
                                        edge_idx[1].unsqueeze(1).expand_as(time_emb),
                                        time_emb)
            merchant_count.scatter_add_(0, edge_idx[1].unsqueeze(1)[:, :1],
                                         torch.ones(edge_idx.shape[1], 1,
                                                    device=h_dict['merchant'].device))
            merchant_count = merchant_count.clamp(min=1)
            h_dict['merchant'] = h_dict['merchant'] + merchant_time / merchant_count

        # Step 3: HGTConv message passing with edge feature injection
        edge_type = ('payer', 'TRANSACTS', 'merchant')
        if edge_attr_dict and edge_type in edge_attr_dict:
            edge_feat_proj = self.edge_proj(edge_attr_dict[edge_type])
        else:
            edge_feat_proj = None

        for i in range(self.n_layers):
            h_residual = {k: v.clone() for k, v in h_dict.items()}

            # Inject edge features into node embeddings before conv
            if edge_feat_proj is not None:
                edge_idx = edge_index_dict[edge_type]
                # Aggregate edge signals to payer (source) nodes
                src_signal = torch.zeros_like(h_dict['payer'])
                src_count = torch.zeros(h_dict['payer'].shape[0], 1,
                                        device=h_dict['payer'].device)
                edge_src = self.edge_to_src(edge_feat_proj)
                src_signal.scatter_add_(0, edge_idx[0].unsqueeze(1).expand_as(edge_src),
                                        edge_src)
                src_count.scatter_add_(0, edge_idx[0].unsqueeze(1)[:, :1],
                                       torch.ones(edge_idx.shape[1], 1,
                                                  device=src_signal.device))
                src_count = src_count.clamp(min=1)
                h_dict['payer'] = h_dict['payer'] + src_signal / src_count

                # Aggregate edge signals to merchant (destination) nodes
                dst_signal = torch.zeros_like(h_dict['merchant'])
                dst_count = torch.zeros(h_dict['merchant'].shape[0], 1,
                                        device=h_dict['merchant'].device)
                edge_dst = self.edge_to_dst(edge_feat_proj)
                dst_signal.scatter_add_(0, edge_idx[1].unsqueeze(1).expand_as(edge_dst),
                                        edge_dst)
                dst_count.scatter_add_(0, edge_idx[1].unsqueeze(1)[:, :1],
                                       torch.ones(edge_idx.shape[1], 1,
                                                  device=dst_signal.device))
                dst_count = dst_count.clamp(min=1)
                h_dict['merchant'] = h_dict['merchant'] + dst_signal / dst_count

            h_dict = self.convs[i](h_dict, edge_index_dict)
            # Residual + LayerNorm + Dropout
            for nt in h_dict:
                h_dict[nt] = self.norms[i][nt](h_dict[nt] + h_residual[nt])
                h_dict[nt] = self.dropout(h_dict[nt])

        # Step 4: Edge classification
        edge_idx = edge_index_dict[edge_type]
        src_emb = h_dict['payer'][edge_idx[0]]   # [E, d_model]
        dst_emb = h_dict['merchant'][edge_idx[1]] # [E, d_model]

        # Use original edge features for classifier
        edge_feat = edge_attr_dict[edge_type] if edge_attr_dict else \
            torch.zeros(edge_idx.shape[1], 12, device=src_emb.device)

        gnn_logits = self.classifier(src_emb, edge_feat, dst_emb)

        # Step 5: Edge anomaly head (bypasses GNN)
        anomaly_logits = self.anomaly_head(edge_feat)

        # Step 6: Gated fusion — learned blend of GNN + edge-only scores
        g = torch.sigmoid(self.gate)  # 0→1, starts at 0.5
        logits = g * gnn_logits + (1 - g) * anomaly_logits

        return logits

    def get_node_embeddings(self, x_dict, edge_index_dict):
        # Get final node embeddings (for XAI visualization).
        h_dict = {
            'payer': self.payer_encoder(x_dict['payer']),
            'merchant': self.merchant_encoder(x_dict['merchant']),
        }
        for i in range(self.n_layers):
            h_residual = {k: v.clone() for k, v in h_dict.items()}
            h_dict = self.convs[i](h_dict, edge_index_dict)
            for nt in h_dict:
                h_dict[nt] = self.norms[i][nt](h_dict[nt] + h_residual[nt])
        return h_dict
