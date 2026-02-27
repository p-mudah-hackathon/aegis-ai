import os
import json
import torch

from core.model import AegisHTGNN
from core.config import PROCESSED_DIR, CHECKPOINT_DIR, EXPORT_DIR


def load_trained_model():
    # Load trained model from checkpoint.
    data = torch.load(os.path.join(PROCESSED_DIR, 'aegis_hetero_data.pt'),
                      weights_only=False)
    with open(os.path.join(PROCESSED_DIR, 'preprocessing_meta.json')) as f:
        meta = json.load(f)

    ckpt = torch.load(os.path.join(CHECKPOINT_DIR, 'best_model.pt'),
                      weights_only=False, map_location='cpu')

    graph_meta = data.metadata()
    args = ckpt.get('args', {})

    model = AegisHTGNN(
        meta=graph_meta,
        cat_sizes=meta['cat_sizes'],
        edge_feat_dim=meta['edge_feat_dim'],
        d_model=args.get('d_model', 64),
        n_heads=args.get('n_heads', 4),
        n_layers=args.get('n_layers', 3),
        dropout=0.0,
    )
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    return model, data, meta, ckpt


def export_state_dict(model, meta, ckpt):
    # Export as state_dict + architecture config (safest for PAI-EAS).
    os.makedirs(EXPORT_DIR, exist_ok=True)

    # Save model weights
    weights_path = os.path.join(EXPORT_DIR, 'aegis_htgnn_weights.pt')
    torch.save(model.state_dict(), weights_path)
    print(f"  Weights saved: {weights_path}")

    # Save model architecture config
    args = ckpt.get('args', {})
    model_config = {
        'model_class': 'AegisHTGNN',
        'cat_sizes': meta['cat_sizes'],
        'edge_feat_dim': meta['edge_feat_dim'],
        'd_model': args.get('d_model', 64),
        'n_heads': args.get('n_heads', 4),
        'n_layers': args.get('n_layers', 3),
        'dropout': 0.0,
        'training_epoch': ckpt.get('epoch', -1),
        'val_auprc': ckpt.get('val_auprc', -1),
    }
    config_path = os.path.join(EXPORT_DIR, 'model_config.json')
    with open(config_path, 'w') as f:
        json.dump(model_config, f, indent=2)
    print(f"  Config saved: {config_path}")

    return weights_path, config_path


def export_inference_config(meta):
    # Save preprocessing configuration for inference.
    inference_config = {
        'preprocessing': {
            'label_encoder_classes': meta['label_encoder_classes'],
            'norm_params': meta['norm_params'],
            'cat_sizes': meta['cat_sizes'],
            'edge_feat_dim': meta['edge_feat_dim'],
        },
        'edge_feature_order': [
            'amount_idr', 'original_amount',
            'hour_sin', 'hour_cos',
            'dow_sin', 'dow_cos',
            'amount_zscore_merchant',
            'payer_txn_count_1h', 'payer_unique_merchant_1h',
            'merchant_txn_velocity_1h',
            'time_since_last_txn_sec', 'is_first_txn_payer',
        ],
        'node_types': ['payer', 'merchant'],
        'edge_type': ['payer', 'TRANSACTS', 'merchant'],
        'payer_features': ['source_issuer_idx', 'source_country_idx'],
        'merchant_features': ['merchant_mcc_idx', 'merchant_location_idx',
                              'qris_type_idx'],
    }

    # Load threshold from evaluation
    eval_path = os.path.join(CHECKPOINT_DIR, 'evaluation_results.json')
    if os.path.exists(eval_path):
        with open(eval_path) as f:
            eval_results = json.load(f)
        inference_config['fraud_threshold'] = eval_results.get('best_threshold', 0.5)
    else:
        inference_config['fraud_threshold'] = 0.5

    config_path = os.path.join(EXPORT_DIR, 'inference_config.json')
    with open(config_path, 'w') as f:
        json.dump(inference_config, f, indent=2)
    print(f"  Inference config saved: {config_path}")

    return config_path


def validate_export(model, data):
    # Validate exported model matches original outputs.
    edge_type = ('payer', 'TRANSACTS', 'merchant')

    with torch.no_grad():
        x_dict = {k: data[k].x for k in data.node_types}
        edge_index_dict = {et: data[et].edge_index for et in data.edge_types}
        edge_attr_dict = {et: data[et].edge_attr for et in data.edge_types
                          if hasattr(data[et], 'edge_attr')}
        timestamps = data[edge_type].timestamps

        original_output = model(x_dict, edge_index_dict, edge_attr_dict,
                                timestamps)

    # Load exported weights and compare
    exported_model = AegisHTGNN(
        meta=data.metadata(),
        cat_sizes=json.load(open(os.path.join(EXPORT_DIR, 'model_config.json')))['cat_sizes'],
        edge_feat_dim=12,
        d_model=64,
        n_heads=4,
        n_layers=3,
        dropout=0.0,
    )
    exported_model.load_state_dict(
        torch.load(os.path.join(EXPORT_DIR, 'aegis_htgnn_weights.pt'),
                    weights_only=True))
    exported_model.eval()

    with torch.no_grad():
        exported_output = exported_model(x_dict, edge_index_dict,
                                          edge_attr_dict, timestamps)

    # Compare
    diff = (original_output - exported_output).abs().max().item()
    print(f"\n  Validation: max output diff = {diff:.2e}")
    if diff < 1e-5:
        print("  Validation: PASSED (outputs match)")
    else:
        print(f"  WARNING: Output difference {diff:.2e} exceeds tolerance 1e-5")

    return diff < 1e-5


def main():
    print("=" * 60)
    print("AegisNode Model Export")
    print("=" * 60)

    model, data, meta, ckpt = load_trained_model()
    print(f"  Loaded model from epoch {ckpt.get('epoch', '?')}")
    print(f"  Val AUPRC: {ckpt.get('val_auprc', '?')}")

    # Export weights + config
    print("\n[1/3] Exporting weights and config...")
    export_state_dict(model, meta, ckpt)

    # Export inference config
    print("\n[2/3] Exporting inference config...")
    export_inference_config(meta)

    # Validate
    print("\n[3/3] Validating export...")
    passed = validate_export(model, data)

    print("\n" + "=" * 60)
    if passed:
        print("  EXPORT COMPLETE - Ready for PAI-EAS deployment")
        print(f"\n  Exported files in: {EXPORT_DIR}/")
        print("    - aegis_htgnn_weights.pt    (model weights)")
        print("    - model_config.json          (architecture config)")
        print("    - inference_config.json      (preprocessing + threshold)")
    else:
        print("  WARNING: Validation failed. Check export.")
    print("=" * 60)
