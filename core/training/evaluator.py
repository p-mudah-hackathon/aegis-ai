import os
import json
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    average_precision_score, roc_auc_score, f1_score,
    precision_score, recall_score, confusion_matrix,
    classification_report, precision_recall_curve
)

from core.model import AegisHTGNN
from core.config import PROCESSED_DIR, CHECKPOINT_DIR


def load_model_and_data():
    # Load best checkpoint, data, and metadata.
    # Load data
    data = torch.load(os.path.join(PROCESSED_DIR, 'aegis_hetero_data.pt'),
                      weights_only=False)
    with open(os.path.join(PROCESSED_DIR, 'preprocessing_meta.json')) as f:
        meta = json.load(f)

    # Load checkpoint
    ckpt_path = os.path.join(CHECKPOINT_DIR, 'best_model.pt')
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"No checkpoint found at {ckpt_path}. Run train.py first.")

    ckpt = torch.load(ckpt_path, weights_only=False, map_location='cpu')

    # Reconstruct model
    graph_meta = data.metadata()
    args = ckpt.get('args', {})
    model = AegisHTGNN(
        meta=graph_meta,
        cat_sizes=meta['cat_sizes'],
        edge_feat_dim=meta['edge_feat_dim'],
        d_model=args.get('d_model', 64),
        n_heads=args.get('n_heads', 4),
        n_layers=args.get('n_layers', 3),
        dropout=args.get('dropout', 0.3),
    )
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    print(f"  Loaded checkpoint from epoch {ckpt['epoch']}")
    print(f"  Val AUPRC at checkpoint: {ckpt['val_auprc']:.4f}")

    return model, data, meta


@torch.no_grad()
def run_evaluation():
    # Full evaluation on test set.
    print("=" * 60)
    print("AegisNode Model Evaluation")
    print("=" * 60)

    device = torch.device('cpu')  # Evaluation on CPU is fine
    model, data, meta = load_model_and_data()
    model = model.to(device)

    edge_type = ('payer', 'TRANSACTS', 'merchant')

    # Forward pass
    x_dict = {k: data[k].x.to(device) for k in data.node_types}
    edge_index_dict = {et: data[et].edge_index.to(device)
                       for et in data.edge_types}
    edge_attr_dict = {et: data[et].edge_attr.to(device)
                      for et in data.edge_types if hasattr(data[et], 'edge_attr')}
    timestamps = data[edge_type].timestamps.to(device)

    logits = model(x_dict, edge_index_dict, edge_attr_dict, timestamps)
    probs = torch.sigmoid(logits).cpu().numpy()

    # ============================================================
    # TEST SET EVALUATION
    # ============================================================
    for split_name in ['val', 'test']:
        mask = getattr(data[edge_type], f'{split_name}_mask').cpu().numpy()
        y_true = data[edge_type].y.cpu().numpy()[mask]
        y_scores = probs[mask]

        print(f"\n{'=' * 60}")
        print(f"  {split_name.upper()} SET RESULTS")
        print(f"{'=' * 60}")
        print(f"  Samples: {len(y_true)}, Fraud: {int(y_true.sum())}")

        if len(np.unique(y_true)) < 2:
            print(f"  WARNING: Only one class in {split_name} set, skipping metrics.")
            continue

        # Overall metrics
        auprc = average_precision_score(y_true, y_scores)
        auroc = roc_auc_score(y_true, y_scores)

        print(f"\n  AUPRC  : {auprc:.4f} (random baseline: {y_true.mean():.4f})")
        print(f"  AUROC  : {auroc:.4f}")

        # Optimal threshold via F1
        precisions, recalls, thresholds = precision_recall_curve(y_true, y_scores)
        f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-8)
        best_idx = np.argmax(f1_scores)
        best_thresh = thresholds[best_idx] if best_idx < len(thresholds) else 0.5

        y_pred = (y_scores >= best_thresh).astype(int)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        prec = precision_score(y_true, y_pred, zero_division=0)
        rec = recall_score(y_true, y_pred, zero_division=0)

        print(f"\n  Optimal threshold: {best_thresh:.3f}")
        print(f"  F1        : {f1:.4f}")
        print(f"  Precision : {prec:.4f}")
        print(f"  Recall    : {rec:.4f}")

        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        print(f"\n  Confusion Matrix (threshold={best_thresh:.3f}):")
        print(f"               Predicted")
        print(f"              Normal  Fraud")
        print(f"  Actual Normal  {cm[0, 0]:>6d}  {cm[0, 1]:>5d}")
        print(f"  Actual Fraud   {cm[1, 0]:>6d}  {cm[1, 1]:>5d}")

        tn, fp, fn, tp = cm.ravel()
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        print(f"\n  FPR (False Positive Rate): {fpr:.4f}")
        print(f"  TP: {tp}, FP: {fp}, TN: {tn}, FN: {fn}")

    # ============================================================
    # PER-FRAUD-TYPE RECALL (on full dataset)
    # ============================================================
    print(f"\n{'=' * 60}")
    print("  PER-FRAUD-TYPE RECALL")
    print(f"{'=' * 60}")

    test_mask = data[edge_type].test_mask.cpu().numpy()
    val_mask = data[edge_type].val_mask.cpu().numpy()
    eval_mask = test_mask | val_mask  # Combined for more fraud samples

    fraud_types = data[edge_type].fraud_types
    y_true_all = data[edge_type].y.cpu().numpy()

    # Use the optimal threshold from validation set
    y_pred_all = (probs >= best_thresh).astype(int)

    # Only evaluate fraud samples in eval set
    eval_fraud_mask = eval_mask & (y_true_all == 1)
    eval_fraud_indices = np.where(eval_fraud_mask)[0]

    if len(eval_fraud_indices) > 0:
        fraud_type_list = fraud_types[eval_fraud_indices]
        fraud_preds = y_pred_all[eval_fraud_indices]

        unique_types = np.unique(fraud_type_list)
        print(f"\n  {'Fraud Type':<20s} {'Total':>6s} {'Detected':>8s} {'Recall':>8s}")
        print(f"  {'-' * 45}")

        for ft in unique_types:
            if ft == 'normal' or pd.isna(ft) if hasattr(ft, '__class__') and 'float' in ft.__class__.__name__ else False:
                continue
            ft_mask = fraud_type_list == ft
            total = ft_mask.sum()
            detected = fraud_preds[ft_mask].sum()
            rec = detected / total if total > 0 else 0
            print(f"  {ft:<20s} {total:>6d} {detected:>8d} {rec:>8.2%}")

    # Save evaluation results
    results = {
        'best_threshold': float(best_thresh),
        'metrics': {
            'auprc': float(auprc),
            'auroc': float(auroc),
            'f1': float(f1),
            'precision': float(prec),
            'recall': float(rec),
        },
    }
    results_path = os.path.join(CHECKPOINT_DIR, 'evaluation_results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved: {results_path}")
    print("=" * 60)
