import os
import json
import argparse
import numpy as np
import torch
import pandas as pd

from core.model import AegisHTGNN
from core.config import PROCESSED_DIR, CHECKPOINT_DIR, EXPLANATIONS_DIR, CSV_PATH


# Edge feature names in order (must match data_preprocessing.py)
EDGE_FEATURE_NAMES = [
    'amount_idr', 'original_amount',
    'hour_sin', 'hour_cos',
    'dow_sin', 'dow_cos',
    'amount_zscore_merchant',
    'payer_txn_count_1h', 'payer_unique_merchant_1h',
    'merchant_txn_velocity_1h',
    'time_since_last_txn_sec', 'is_first_txn_payer',
]

# Human-readable names for dashboard
FEATURE_DISPLAY_NAMES = {
    'amount_idr': 'Transaction Amount',
    'original_amount': 'Original Amount (Foreign)',
    'hour_sin': 'Time of Day Pattern',
    'hour_cos': 'Time of Day Pattern',
    'dow_sin': 'Day of Week Pattern',
    'dow_cos': 'Day of Week Pattern',
    'amount_zscore_merchant': 'Unusual Amount for Merchant',
    'payer_txn_count_1h': 'Rapid Transaction Count',
    'payer_unique_merchant_1h': 'Multiple Merchants Visited',
    'merchant_txn_velocity_1h': 'Merchant Transaction Spike',
    'time_since_last_txn_sec': 'Time Since Last Transaction',
    'is_first_txn_payer': 'New Payer (First Transaction)',
}


def parse_args():
    p = argparse.ArgumentParser(description='AegisNode XAI Explanations')
    p.add_argument('--top-k', type=int, default=20,
                   help='Number of transactions to explain')
    p.add_argument('--txn-id', type=str, default=None,
                   help='Specific transaction ID to explain')
    p.add_argument('--threshold', type=float, default=None,
                   help='Fraud threshold (auto-loaded from eval if available)')
    p.add_argument('--mode', type=str, default='flagged_fraud',
                   choices=['flagged_fraud', 'all_flagged', 'top_k'],
                   help='Selection mode: flagged_fraud=TPs only (default), '
                        'all_flagged=all above threshold, top_k=highest scores')
    return p.parse_args()


def load_components():
    # Load model, data, and original CSV for transaction IDs.
    # Load data
    data = torch.load(os.path.join(PROCESSED_DIR, 'aegis_hetero_data.pt'),
                      weights_only=False)
    with open(os.path.join(PROCESSED_DIR, 'preprocessing_meta.json')) as f:
        meta = json.load(f)

    # Load checkpoint
    ckpt = torch.load(os.path.join(CHECKPOINT_DIR, 'best_model.pt'),
                      weights_only=False, map_location='cpu')

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
        dropout=0.0,  # No dropout during inference
    )
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    # Load original CSV for txn_ids and readable values
    df = pd.read_csv(CSV_PATH, parse_dates=['timestamp'])
    df = df.sort_values('timestamp').reset_index(drop=True)

    # Load threshold
    threshold = 0.5
    eval_results_path = os.path.join(CHECKPOINT_DIR, 'evaluation_results.json')
    if os.path.exists(eval_results_path):
        with open(eval_results_path) as f:
            eval_results = json.load(f)
        threshold = eval_results.get('best_threshold', 0.5)

    return model, data, meta, df, threshold


def gradient_feature_importance(model, data, edge_idx_target):
    # 
    #     Compute feature importance via input gradient method.
    #     Faster and more stable than GNNExplainer for edge classification.
    #     
    edge_type = ('payer', 'TRANSACTS', 'merchant')

    x_dict = {k: data[k].x for k in data.node_types}
    edge_index_dict = {et: data[et].edge_index for et in data.edge_types}

    # Make edge features require gradient
    edge_attr = data[edge_type].edge_attr.clone().detach().requires_grad_(True)
    edge_attr_dict = {edge_type: edge_attr}
    timestamps = data[edge_type].timestamps

    # Forward pass
    logits = model(x_dict, edge_index_dict, edge_attr_dict, timestamps)

    # Backward from target edge
    target_logit = logits[edge_idx_target]
    target_logit.backward()

    # Gradient * input = importance
    grad = edge_attr.grad[edge_idx_target]
    importance = (grad * edge_attr[edge_idx_target].detach()).abs()

    return importance.detach().numpy()


def find_related_transactions(data, df, edge_idx, n_hops=2):
    # Find related transactions via graph neighborhood.
    edge_type = ('payer', 'TRANSACTS', 'merchant')
    edge_index = data[edge_type].edge_index.numpy()

    # Get payer and merchant for the target edge
    payer_node = edge_index[0, edge_idx]
    merchant_node = edge_index[1, edge_idx]

    # Find all edges connected to same payer or merchant
    same_payer = np.where(edge_index[0] == payer_node)[0]
    same_merchant = np.where(edge_index[1] == merchant_node)[0]
    related_indices = np.union1d(same_payer, same_merchant)

    # Remove the target edge itself
    related_indices = related_indices[related_indices != edge_idx]

    # Get corresponding transaction IDs
    related_txn_ids = df.iloc[related_indices]['txn_id'].tolist()

    # Limit to nearest in time (sort by temporal distance)
    target_ts = df.iloc[edge_idx]['timestamp']
    related_df = df.iloc[related_indices].copy()
    related_df['time_dist'] = abs((related_df['timestamp'] - target_ts).dt.total_seconds())
    related_df = related_df.sort_values('time_dist').head(10)

    return related_df['txn_id'].tolist()


def explain_transaction(model, data, df, edge_idx, threshold):
    # Generate explanation for a single transaction.
    edge_type = ('payer', 'TRANSACTS', 'merchant')

    # Get fraud score
    with torch.no_grad():
        x_dict = {k: data[k].x for k in data.node_types}
        edge_index_dict = {et: data[et].edge_index for et in data.edge_types}
        edge_attr_dict = {et: data[et].edge_attr for et in data.edge_types
                          if hasattr(data[et], 'edge_attr')}
        timestamps = data[edge_type].timestamps
        logits = model(x_dict, edge_index_dict, edge_attr_dict, timestamps)
        fraud_score = torch.sigmoid(logits[edge_idx]).item()

    # Get feature importance
    importance = gradient_feature_importance(model, data, edge_idx)

    # Build top reasons
    row = df.iloc[edge_idx]
    feature_scores = []
    for i, feat_name in enumerate(EDGE_FEATURE_NAMES):
        # Merge sin/cos pairs
        if feat_name in ('hour_cos', 'dow_cos'):
            continue  # Merged with sin

        imp = float(importance[i])
        if feat_name == 'hour_sin':
            imp += float(importance[i + 1])  # Add cos component
            feat_name = 'hour_of_day'
            raw_value = int(row['hour_of_day'])
        elif feat_name == 'dow_sin':
            imp += float(importance[i + 1])
            feat_name = 'day_of_week'
            raw_value = int(row['day_of_week'])
        else:
            # Get raw value from original data
            if feat_name in df.columns:
                raw_value = row[feat_name]
                if isinstance(raw_value, (np.integer, np.int64)):
                    raw_value = int(raw_value)
                elif isinstance(raw_value, (np.floating, np.float64)):
                    raw_value = round(float(raw_value), 2)
            else:
                raw_value = float(data[edge_type].edge_attr[edge_idx, i])

        display_name = FEATURE_DISPLAY_NAMES.get(
            feat_name, feat_name.replace('_', ' ').title())

        feature_scores.append({
            'feature': feat_name,
            'display_name': display_name,
            'importance': round(imp, 4),
            'value': raw_value,
        })

    # Sort by importance, take top 5
    feature_scores.sort(key=lambda x: x['importance'], reverse=True)
    top_reasons = feature_scores[:5]

    # Normalize importance to sum to 1
    total_imp = sum(r['importance'] for r in top_reasons) or 1
    for r in top_reasons:
        r['importance'] = round(r['importance'] / total_imp, 4)

    # Find related transactions
    related_txns = find_related_transactions(data, df, edge_idx)

    # Build explanation
    explanation = {
        'txn_id': str(row['txn_id']),
        'timestamp': str(row['timestamp']),
        'fraud_score': round(fraud_score, 4),
        'is_flagged': fraud_score >= threshold,
        'threshold': round(threshold, 4),
        'payer_id': str(row['payer_token_id']),
        'merchant_id': str(row['merchant_nmid']),
        'amount_idr': int(row['amount_idr']),
        'merchant_location': str(row['merchant_location']),
        'top_reasons': top_reasons,
        'related_transactions': related_txns[:5],
        'is_fraud_actual': int(row['is_fraud']),
        'fraud_type': str(row.get('fraud_type', '')),
    }

    return explanation


def main():
    args = parse_args()
    print("=" * 60)
    print("AegisNode XAI Explanation Generator")
    print("=" * 60)

    model, data, meta, df, threshold = load_components()
    if args.threshold is not None:
        threshold = args.threshold
    print(f"  Fraud threshold: {threshold:.3f}")

    edge_type = ('payer', 'TRANSACTS', 'merchant')

    # Determine which transactions to explain
    if args.txn_id:
        # Specific transaction
        idx = df[df['txn_id'] == args.txn_id].index
        if len(idx) == 0:
            print(f"  ERROR: Transaction {args.txn_id} not found.")
            return
        target_indices = idx.tolist()
        print(f"  Explaining transaction: {args.txn_id}")
    else:
        # Score all transactions first
        with torch.no_grad():
            x_dict = {k: data[k].x for k in data.node_types}
            edge_index_dict = {et: data[et].edge_index for et in data.edge_types}
            edge_attr_dict = {et: data[et].edge_attr for et in data.edge_types
                              if hasattr(data[et], 'edge_attr')}
            timestamps = data[edge_type].timestamps
            logits = model(x_dict, edge_index_dict, edge_attr_dict, timestamps)
            scores = torch.sigmoid(logits).numpy()

        y_true = data[edge_type].y.numpy()

        if args.mode == 'flagged_fraud':
            # True Positives: flagged AND actually fraud (best for demo/dashboard)
            flagged_mask = scores >= threshold
            fraud_mask = y_true == 1
            tp_mask = flagged_mask & fraud_mask
            tp_indices = np.where(tp_mask)[0]
            sorted_tp = tp_indices[np.argsort(scores[tp_indices])[::-1]]
            target_indices = sorted_tp[:args.top_k].tolist()
            print(f"  Mode: flagged_fraud — {len(tp_indices)} TPs found, "
                  f"explaining top {len(target_indices)}")
        elif args.mode == 'all_flagged':
            # All flagged transactions (TPs + FPs, for debugging)
            flagged_indices = np.where(scores >= threshold)[0]
            sorted_flagged = flagged_indices[np.argsort(scores[flagged_indices])[::-1]]
            target_indices = sorted_flagged[:args.top_k].tolist()
            print(f"  Mode: all_flagged — {len(flagged_indices)} flagged, "
                  f"explaining top {len(target_indices)}")
        else:
            # Original: top-K by raw score regardless of labels
            top_indices = np.argsort(scores)[::-1][:args.top_k]
            target_indices = top_indices.tolist()
            print(f"  Mode: top_k — explaining top {len(target_indices)} by score")

    # Generate explanations
    explanations = []
    for i, idx in enumerate(target_indices):
        print(f"  [{i + 1}/{len(target_indices)}] Explaining edge {idx}...",
              end='', flush=True)
        exp = explain_transaction(model, data, df, idx, threshold)
        explanations.append(exp)
        print(f" score={exp['fraud_score']:.4f} "
              f"{'FRAUD' if exp['is_fraud_actual'] else 'normal'}")

    # Save
    os.makedirs(EXPLANATIONS_DIR, exist_ok=True)
    output_path = os.path.join(EXPLANATIONS_DIR, 'fraud_explanations.json')
    with open(output_path, 'w') as f:
        json.dump(explanations, f, indent=2, default=str)
    print(f"\n  Saved {len(explanations)} explanations to: {output_path}")

    # Print sample
    if explanations:
        print("\n" + "=" * 60)
        print("  SAMPLE EXPLANATION")
        print("=" * 60)
        sample = explanations[0]
        print(f"  Transaction: {sample['txn_id']}")
        print(f"  Fraud Score: {sample['fraud_score']:.4f}")
        print(f"  Flagged: {sample['is_flagged']}")
        print(f"  Amount: {sample['amount_idr']:,} IDR")
        print(f"  Top Reasons:")
        for r in sample['top_reasons']:
            print(f"    - {r['display_name']}: {r['value']} "
                  f"(importance: {r['importance']:.2%})")
        print(f"  Related transactions: {sample['related_transactions'][:3]}")

    print("=" * 60)
