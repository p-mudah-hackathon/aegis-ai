import os
import json
import numpy as np
import pandas as pd
import torch
from torch_geometric.data import HeteroData
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

from core.config import CSV_PATH, PROCESSED_DIR, TRAIN_END, VAL_END


# ============================================================
# FEATURE ENGINEERING
# ============================================================
def cyclical_encode(values, max_val):
    # Encode cyclic features (hour, day) as sin/cos pair.
    angle = 2 * np.pi * values / max_val
    return np.sin(angle), np.cos(angle)


def log_transform_safe(values, epsilon=1.0):
    # Log-transform with safe handling of -1 sentinel and zeros.
    v = values.copy().astype(float)
    v[v < 0] = 0  # Replace -1 sentinel with 0
    return np.log1p(v + epsilon)


def normalize(values):
    # Min-max normalization to [0, 1].
    v = values.astype(float)
    vmin, vmax = v.min(), v.max()
    if vmax - vmin == 0:
        return np.zeros_like(v)
    return (v - vmin) / (vmax - vmin)


# ============================================================
# MAIN PREPROCESSING
# ============================================================
class AegisPreprocessor:
    def __init__(self, csv_path=None, output_dir=None):
        self.csv_path = csv_path or CSV_PATH
        self.output_dir = output_dir or PROCESSED_DIR
        self.label_encoders = {}
        self.norm_params = {}

    def _load_and_encode(self):
        # Load CSV and encode categorical features.
        print("[1/5] Loading data...")
        df = pd.read_csv(self.csv_path, parse_dates=['timestamp'])
        df = df.sort_values('timestamp').reset_index(drop=True)
        print(f"  Loaded {len(df)} transactions")

        # Label encode categoricals
        print("[2/5] Encoding features...")
        cat_cols = {
            'source_issuer': None,
            'source_country': None,
            'merchant_mcc': None,
            'merchant_location': None,
            'qris_type': None,
        }
        for col in cat_cols:
            le = LabelEncoder()
            df[f'{col}_enc'] = le.fit_transform(df[col].astype(str))
            self.label_encoders[col] = le
            cat_cols[col] = len(le.classes_)
            print(f"  {col}: {len(le.classes_)} classes")

        # Encode payer_token_id and merchant_nmid as node indices
        payer_le = LabelEncoder()
        df['payer_idx'] = payer_le.fit_transform(df['payer_token_id'])
        self.label_encoders['payer_token_id'] = payer_le
        n_payers = len(payer_le.classes_)

        merchant_le = LabelEncoder()
        df['merchant_idx'] = merchant_le.fit_transform(df['merchant_nmid'])
        self.label_encoders['merchant_nmid'] = merchant_le
        n_merchants = len(merchant_le.classes_)

        print(f"  Payer nodes: {n_payers}, Merchant nodes: {n_merchants}")
        return df, n_payers, n_merchants, cat_cols

    def _build_node_features(self, df, n_payers, n_merchants):
        # Build static feature matrices for payer and merchant nodes.
        print("[3/5] Building node features...")

        # Payer features: [issuer_enc, country_enc]
        # Get the most common issuer/country per payer
        payer_info = df.groupby('payer_idx').agg({
            'source_issuer_enc': 'first',
            'source_country_enc': 'first',
        }).reset_index().sort_values('payer_idx')

        payer_x = torch.zeros(n_payers, 2, dtype=torch.long)
        payer_x[:, 0] = torch.tensor(payer_info['source_issuer_enc'].values)
        payer_x[:, 1] = torch.tensor(payer_info['source_country_enc'].values)

        # Merchant features: [mcc_enc, location_enc, qris_type_enc]
        merchant_info = df.groupby('merchant_idx').agg({
            'merchant_mcc_enc': 'first',
            'merchant_location_enc': 'first',
            'qris_type_enc': 'first',
        }).reset_index().sort_values('merchant_idx')

        merchant_x = torch.zeros(n_merchants, 3, dtype=torch.long)
        merchant_x[:, 0] = torch.tensor(merchant_info['merchant_mcc_enc'].values)
        merchant_x[:, 1] = torch.tensor(merchant_info['merchant_location_enc'].values)
        merchant_x[:, 2] = torch.tensor(merchant_info['qris_type_enc'].values)

        print(f"  Payer features: {payer_x.shape}")
        print(f"  Merchant features: {merchant_x.shape}")
        return payer_x, merchant_x

    def _build_edge_features(self, df):
        # Build edge index, features, labels, and timestamps.
        print("[4/5] Building edge features...")

        # Edge index: [src, dst] = [payer_idx, merchant_idx]
        edge_index = torch.tensor(
            np.stack([df['payer_idx'].values, df['merchant_idx'].values]),
            dtype=torch.long
        )

        # Edge features: continuous features normalized
        # 1. amount_idr (log-normalized)
        amt_log = log_transform_safe(df['amount_idr'].values)
        amt_norm = normalize(amt_log)
        self.norm_params['amount_idr'] = {
            'min': float(amt_log.min()), 'max': float(amt_log.max())}

        # 2. original_amount (log-normalized)
        orig_log = log_transform_safe(df['original_amount'].values)
        orig_norm = normalize(orig_log)
        self.norm_params['original_amount'] = {
            'min': float(orig_log.min()), 'max': float(orig_log.max())}

        # 3-4. hour_of_day (cyclical sin/cos)
        hour_sin, hour_cos = cyclical_encode(df['hour_of_day'].values, 24)

        # 5-6. day_of_week (cyclical sin/cos)
        dow_sin, dow_cos = cyclical_encode(df['day_of_week'].values, 7)

        # 7. amount_zscore_merchant (clip to [-5, 5] and normalize)
        zscore = np.clip(df['amount_zscore_merchant'].values, -5, 5)
        zscore_norm = (zscore + 5) / 10  # Map [-5,5] to [0,1]

        # 8. payer_txn_count_1h (log-normalized)
        ptc = log_transform_safe(df['payer_txn_count_1h'].values)
        ptc_norm = normalize(ptc)

        # 9. payer_unique_merchant_1h (log-normalized)
        pum = log_transform_safe(df['payer_unique_merchant_1h'].values)
        pum_norm = normalize(pum)

        # 10. merchant_txn_velocity_1h (log-normalized)
        mtv = log_transform_safe(df['merchant_txn_velocity_1h'].values)
        mtv_norm = normalize(mtv)

        # 11. time_since_last_txn_sec (log-normalized, -1→0)
        tsl = log_transform_safe(df['time_since_last_txn_sec'].values)
        tsl_norm = normalize(tsl)

        # 12. is_first_txn_payer (binary)
        is_first = df['is_first_txn_payer'].values.astype(float)

        # Stack all edge features: [n_edges, 12]
        edge_attr = torch.tensor(np.stack([
            amt_norm, orig_norm,
            hour_sin, hour_cos,
            dow_sin, dow_cos,
            zscore_norm,
            ptc_norm, pum_norm, mtv_norm,
            tsl_norm, is_first,
        ], axis=1), dtype=torch.float32)

        # Labels
        edge_label = torch.tensor(df['is_fraud'].values, dtype=torch.float32)

        # Timestamps (epoch seconds for temporal encoding)
        timestamps = torch.tensor(
            df['timestamp'].astype(np.int64).values // 10**9,
            dtype=torch.float64
        )

        # Fraud type (for evaluation only, not training)
        fraud_types = df['fraud_type'].values

        print(f"  Edge index: {edge_index.shape}")
        print(f"  Edge features: {edge_attr.shape} (12 dims)")
        print(f"  Edge labels: {edge_label.shape} (fraud={edge_label.sum().int()})")

        return edge_index, edge_attr, edge_label, timestamps, fraud_types

    def _temporal_split(self, df, timestamps, edge_index, edge_attr,
                        edge_label, fraud_types):
        # Split by time — NO data leakage.
        print("[5/5] Temporal splitting...")

        train_end = pd.Timestamp(TRAIN_END).timestamp()
        val_end = pd.Timestamp(VAL_END).timestamp()
        ts_np = timestamps.numpy()

        train_mask = ts_np <= train_end
        val_mask = (ts_np > train_end) & (ts_np <= val_end)
        test_mask = ts_np > val_end

        splits = {}
        for name, mask in [('train', train_mask), ('val', val_mask), ('test', test_mask)]:
            idx = np.where(mask)[0]
            fraud_count = int(edge_label[idx].sum())
            total = len(idx)
            splits[name] = {
                'indices': torch.tensor(idx, dtype=torch.long),
                'n_total': total,
                'n_fraud': fraud_count,
                'fraud_rate': fraud_count / total * 100 if total > 0 else 0,
            }
            print(f"  {name:5s}: {total:>6,d} txns, {fraud_count:>3d} fraud ({splits[name]['fraud_rate']:.3f}%)")

        # Verify temporal integrity
        train_max_ts = ts_np[train_mask].max() if train_mask.any() else 0
        val_min_ts = ts_np[val_mask].min() if val_mask.any() else float('inf')
        val_max_ts = ts_np[val_mask].max() if val_mask.any() else 0
        test_min_ts = ts_np[test_mask].min() if test_mask.any() else float('inf')

        assert train_max_ts < val_min_ts, "Temporal leak: train overlaps val!"
        assert val_max_ts < test_min_ts, "Temporal leak: val overlaps test!"
        print("  Temporal integrity: VERIFIED (no leakage)")

        return splits

    def process(self):
        # Main processing pipeline.
        print("=" * 60)
        print("AegisNode Data Preprocessing")
        print("=" * 60)

        # Step 1-2: Load and encode
        df, n_payers, n_merchants, cat_sizes = self._load_and_encode()

        # Step 3: Node features
        payer_x, merchant_x = self._build_node_features(df, n_payers, n_merchants)

        # Step 4: Edge features
        edge_index, edge_attr, edge_label, timestamps, fraud_types = \
            self._build_edge_features(df)

        # Step 5: Temporal split
        splits = self._temporal_split(df, timestamps, edge_index, edge_attr,
                                      edge_label, fraud_types)

        # Build HeteroData object
        data = HeteroData()

        # Node features (categorical indices for embedding layers)
        data['payer'].x = payer_x
        data['payer'].num_nodes = n_payers

        data['merchant'].x = merchant_x
        data['merchant'].num_nodes = n_merchants

        # Edge data — forward direction
        edge_type = ('payer', 'TRANSACTS', 'merchant')
        data[edge_type].edge_index = edge_index
        data[edge_type].edge_attr = edge_attr
        data[edge_type].y = edge_label
        data[edge_type].timestamps = timestamps

        # Reverse edges — required for HGTConv bidirectional message passing
        # Without this, payer nodes never receive messages (only source nodes)
        rev_edge_type = ('merchant', 'rev_TRANSACTS', 'payer')
        data[rev_edge_type].edge_index = edge_index.flip(0)  # Swap src/dst
        data[rev_edge_type].edge_attr = edge_attr.clone()     # Same features

        # Split masks (only on forward edges — these carry the labels)
        for split_name, split_info in splits.items():
            data[edge_type][f'{split_name}_mask'] = torch.zeros(
                edge_label.shape[0], dtype=torch.bool)
            data[edge_type][f'{split_name}_mask'][split_info['indices']] = True

        # Store fraud types as metadata (string array for evaluation)
        data[edge_type].fraud_types = fraud_types

        # Save
        os.makedirs(self.output_dir, exist_ok=True)

        # Save HeteroData
        data_path = os.path.join(self.output_dir, 'aegis_hetero_data.pt')
        torch.save(data, data_path)
        print(f"\n  Saved: {data_path}")

        # Save metadata (label encoders, norm params, cat sizes)
        meta = {
            'n_payers': n_payers,
            'n_merchants': n_merchants,
            'cat_sizes': {
                'source_issuer': int(cat_sizes['source_issuer']),
                'source_country': int(cat_sizes['source_country']),
                'merchant_mcc': int(cat_sizes['merchant_mcc']),
                'merchant_location': int(cat_sizes['merchant_location']),
                'qris_type': int(cat_sizes['qris_type']),
            },
            'edge_feat_dim': 12,
            'norm_params': self.norm_params,
            'splits': {k: {kk: vv for kk, vv in v.items() if kk != 'indices'}
                       for k, v in splits.items()},
            'label_encoder_classes': {
                col: le.classes_.tolist()
                for col, le in self.label_encoders.items()
            },
        }
        meta_path = os.path.join(self.output_dir, 'preprocessing_meta.json')
        with open(meta_path, 'w') as f:
            json.dump(meta, f, indent=2, default=str)
        print(f"  Saved: {meta_path}")

        # Validation summary
        print("\n" + "=" * 60)
        print("HETERODATA SUMMARY")
        print("=" * 60)
        print(f"  Node types: {data.node_types}")
        print(f"  Edge types: {data.edge_types}")
        print(f"  Payer features shape:    {data['payer'].x.shape}")
        print(f"  Merchant features shape: {data['merchant'].x.shape}")
        print(f"  Edge index shape:        {data[edge_type].edge_index.shape}")
        print(f"  Edge attr shape:         {data[edge_type].edge_attr.shape}")
        print(f"  Edge labels shape:       {data[edge_type].y.shape}")
        print(f"  Timestamps shape:        {data[edge_type].timestamps.shape}")
        print(f"\n  Train mask: {data[edge_type].train_mask.sum()} edges")
        print(f"  Val mask:   {data[edge_type].val_mask.sum()} edges")
        print(f"  Test mask:  {data[edge_type].test_mask.sum()} edges")

        # Verify no NaN
        assert not torch.isnan(edge_attr).any(), "NaN in edge features!"
        assert not torch.isnan(edge_label).any(), "NaN in edge labels!"
        print("\n  NaN check: PASSED")
        print("=" * 60)

        return data, meta
