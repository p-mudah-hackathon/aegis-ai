import os
import json
import argparse
import time
import numpy as np
import torch
from sklearn.metrics import (
    average_precision_score, roc_auc_score, f1_score,
    precision_score, recall_score
)

from core.model import AegisHTGNN
from core.model.losses import FocalLoss, WeightedBCELoss
from core.config import PROCESSED_DIR, CHECKPOINT_DIR, DEFAULTS


def parse_args():
    p = argparse.ArgumentParser(description='AegisNode HTGNN Training')
    p.add_argument('--epochs', type=int, default=DEFAULTS['epochs'])
    p.add_argument('--lr', type=float, default=DEFAULTS['lr'])
    p.add_argument('--weight-decay', type=float, default=DEFAULTS['weight_decay'])
    p.add_argument('--d-model', type=int, default=DEFAULTS['d_model'])
    p.add_argument('--n-heads', type=int, default=DEFAULTS['n_heads'])
    p.add_argument('--n-layers', type=int, default=DEFAULTS['n_layers'])
    p.add_argument('--dropout', type=float, default=DEFAULTS['dropout'])
    p.add_argument('--patience', type=int, default=DEFAULTS['patience'])
    p.add_argument('--focal-alpha', type=float, default=DEFAULTS['focal_alpha'])
    p.add_argument('--focal-gamma', type=float, default=DEFAULTS['focal_gamma'])
    p.add_argument('--no-save', action='store_true',
                   help='Skip checkpoint saving')
    return p.parse_args()


# ============================================================
# DATA LOADING
# ============================================================
def load_data():
    # Load preprocessed HeteroData and metadata.
    data_path = os.path.join(PROCESSED_DIR, 'aegis_hetero_data.pt')
    meta_path = os.path.join(PROCESSED_DIR, 'preprocessing_meta.json')

    data = torch.load(data_path, weights_only=False)
    with open(meta_path) as f:
        meta = json.load(f)

    return data, meta


# ============================================================
# METRICS
# ============================================================
def compute_metrics(logits, labels):
    # Compute fraud detection metrics.
    probs = torch.sigmoid(logits).detach().cpu().numpy()
    y_true = labels.detach().cpu().numpy()

    metrics = {}

    # Handle edge case: all same class
    if len(np.unique(y_true)) < 2:
        metrics['auprc'] = 0.0
        metrics['auroc'] = 0.0
        metrics['f1'] = 0.0
        metrics['precision'] = 0.0
        metrics['recall'] = 0.0
        return metrics

    metrics['auprc'] = average_precision_score(y_true, probs)
    metrics['auroc'] = roc_auc_score(y_true, probs)

    # Find optimal threshold via F1
    best_f1, best_thresh = 0, 0.5
    for thresh in np.arange(0.1, 0.9, 0.05):
        preds = (probs >= thresh).astype(int)
        f1 = f1_score(y_true, preds, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = thresh

    preds = (probs >= best_thresh).astype(int)
    metrics['f1'] = f1_score(y_true, preds, zero_division=0)
    metrics['precision'] = precision_score(y_true, preds, zero_division=0)
    metrics['recall'] = recall_score(y_true, preds, zero_division=0)
    metrics['threshold'] = float(best_thresh)

    return metrics


# ============================================================
# TRAINING LOOP
# ============================================================
def train_epoch(model, data, optimizer, focal_loss, bce_loss, edge_type, device):
    # Train one epoch with fraud oversampling and mixed loss.
    model.train()
    optimizer.zero_grad()

    x_dict = {k: data[k].x.to(device) for k in data.node_types}
    edge_index_dict = {et: data[et].edge_index.to(device)
                       for et in data.edge_types}
    edge_attr_dict = {et: data[et].edge_attr.to(device)
                      for et in data.edge_types if hasattr(data[et], 'edge_attr')}
    timestamps = data[edge_type].timestamps.to(device)

    # Forward pass (full graph)
    logits = model(x_dict, edge_index_dict, edge_attr_dict, timestamps)

    # Get train edges
    train_mask = data[edge_type].train_mask.to(device)
    labels = data[edge_type].y.to(device)

    # Fraud oversampling: duplicate fraud indices to balance gradients
    fraud_idx = torch.where(train_mask & (labels == 1))[0]
    normal_idx = torch.where(train_mask & (labels == 0))[0]
    n_fraud = len(fraud_idx)
    n_normal = len(normal_idx)

    if n_fraud > 0:
        oversample_ratio = max(1, n_normal // n_fraud)
        balanced_idx = torch.cat([normal_idx, fraud_idx.repeat(oversample_ratio)])
    else:
        balanced_idx = normal_idx

    # Mixed loss: Focal (handles hard examples) + BCE (stability)
    loss = 0.7 * focal_loss(logits[balanced_idx], labels[balanced_idx]) + \
           0.3 * bce_loss(logits[balanced_idx], labels[balanced_idx])
    loss.backward()

    # Gradient clipping
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()

    # Metrics on train set (original mask, not oversampled)
    with torch.no_grad():
        train_metrics = compute_metrics(logits[train_mask], labels[train_mask])

    return loss.item(), train_metrics


@torch.no_grad()
def evaluate(model, data, focal_loss, bce_loss, edge_type, device, mask_name='val_mask'):
    # Evaluate on val or test set.
    model.eval()

    x_dict = {k: data[k].x.to(device) for k in data.node_types}
    edge_index_dict = {et: data[et].edge_index.to(device)
                       for et in data.edge_types}
    edge_attr_dict = {et: data[et].edge_attr.to(device)
                      for et in data.edge_types if hasattr(data[et], 'edge_attr')}
    timestamps = data[edge_type].timestamps.to(device)

    logits = model(x_dict, edge_index_dict, edge_attr_dict, timestamps)

    mask = getattr(data[edge_type], mask_name).to(device)
    labels = data[edge_type].y.to(device)

    loss = 0.7 * focal_loss(logits[mask], labels[mask]) + \
           0.3 * bce_loss(logits[mask], labels[mask])
    metrics = compute_metrics(logits[mask], labels[mask])

    return loss.item(), metrics


# ============================================================
# MAIN
# ============================================================
def main():
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print("=" * 60)
    print("AegisNode HTGNN Training")
    print("=" * 60)
    print(f"  Device: {device}")
    print(f"  Epochs: {args.epochs}")
    print(f"  LR: {args.lr}, Weight Decay: {args.weight_decay}")
    print(f"  Model: d={args.d_model}, heads={args.n_heads}, "
          f"layers={args.n_layers}, dropout={args.dropout}")
    print(f"  Patience: {args.patience}")

    # Load data
    data, meta = load_data()
    edge_type = ('payer', 'TRANSACTS', 'merchant')

    n_train = data[edge_type].train_mask.sum().item()
    n_val = data[edge_type].val_mask.sum().item()
    n_test = data[edge_type].test_mask.sum().item()
    n_fraud_train = data[edge_type].y[data[edge_type].train_mask].sum().int().item()
    n_fraud_val = data[edge_type].y[data[edge_type].val_mask].sum().int().item()

    print(f"\n  Data: {n_train} train / {n_val} val / {n_test} test")
    print(f"  Fraud: {n_fraud_train} train / {n_fraud_val} val")

    # Build model
    graph_meta = data.metadata()
    model = AegisHTGNN(
        meta=graph_meta,
        cat_sizes=meta['cat_sizes'],
        edge_feat_dim=meta['edge_feat_dim'],
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        dropout=args.dropout,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Parameters: {total_params:,} total, {trainable_params:,} trainable")

    # Loss — Mixed: Focal (hard example focus) + BCE (stability)
    # No pos_weight needed — oversampling in train_epoch handles imbalance
    focal_loss = FocalLoss(alpha=args.focal_alpha, gamma=args.focal_gamma)
    bce_loss = WeightedBCELoss(pos_weight=None)
    print(f"\n  Loss: Mixed (0.7x Focal alpha={args.focal_alpha} + 0.3x BCE)")
    print(f"  Fraud oversampling: enabled (balanced gradients)")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=10
    )

    # Training
    print("\n" + "-" * 60)
    history = {'train_loss': [], 'val_loss': [], 'train_auprc': [],
               'val_auprc': [], 'val_auroc': [], 'val_f1': [],
               'val_recall': [], 'lr': []}
    best_val_auprc = 0
    patience_counter = 0
    best_epoch = 0

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        # Train
        train_loss, train_metrics = train_epoch(
            model, data, optimizer, focal_loss, bce_loss, edge_type, device)

        # Validate
        val_loss, val_metrics = evaluate(
            model, data, focal_loss, bce_loss, edge_type, device, 'val_mask')

        scheduler.step(val_metrics.get('auprc', 0))
        elapsed = time.time() - t0

        # Record history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_auprc'].append(train_metrics.get('auprc', 0))
        history['val_auprc'].append(val_metrics.get('auprc', 0))
        history['val_auroc'].append(val_metrics.get('auroc', 0))
        history['val_f1'].append(val_metrics.get('f1', 0))
        history['val_recall'].append(val_metrics.get('recall', 0))
        history['lr'].append(optimizer.param_groups[0]['lr'])

        # Print progress
        print(f"  Epoch {epoch:3d}/{args.epochs} | "
              f"Loss: {train_loss:.4f}/{val_loss:.4f} | "
              f"AUPRC: {train_metrics.get('auprc', 0):.4f}/"
              f"{val_metrics.get('auprc', 0):.4f} | "
              f"F1: {val_metrics.get('f1', 0):.4f} | "
              f"Recall: {val_metrics.get('recall', 0):.4f} | "
              f"{elapsed:.1f}s"
              + (f" *" if val_metrics.get('auprc', 0) > best_val_auprc else ""))

        # Early stopping on val AUPRC
        if val_metrics.get('auprc', 0) > best_val_auprc:
            best_val_auprc = val_metrics['auprc']
            best_epoch = epoch
            patience_counter = 0

            if not args.no_save:
                os.makedirs(CHECKPOINT_DIR, exist_ok=True)
                ckpt_path = os.path.join(CHECKPOINT_DIR, 'best_model.pt')
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_auprc': best_val_auprc,
                    'val_metrics': val_metrics,
                    'args': vars(args),
                    'meta': meta,
                }, ckpt_path)
        else:
            patience_counter += 1

        if patience_counter >= args.patience:
            print(f"\n  Early stopping at epoch {epoch} "
                  f"(best={best_epoch}, AUPRC={best_val_auprc:.4f})")
            break

    # Save training history
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print(f"  Best epoch: {best_epoch}")
    print(f"  Best val AUPRC: {best_val_auprc:.4f}")

    if not args.no_save:
        os.makedirs(CHECKPOINT_DIR, exist_ok=True)
        hist_path = os.path.join(CHECKPOINT_DIR, 'training_history.json')
        with open(hist_path, 'w') as f:
            json.dump(history, f, indent=2)
        print(f"  History saved: {hist_path}")
        print(f"  Checkpoint saved: {os.path.join(CHECKPOINT_DIR, 'best_model.pt')}")

    print("=" * 60)
    return model, history
