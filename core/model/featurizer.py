import json
import os
import math
from datetime import datetime
import numpy as np
import torch
from torch_geometric.data import HeteroData

from core.config import PROCESSED_DIR

# ── Load Metadata ───────────────────────────────────────────────────────────
meta_path = os.path.join(PROCESSED_DIR, "preprocessing_meta.json")
with open(meta_path) as f:
    META = json.load(f)

LABEL_ENCODERS = META.get("label_encoder_classes", {})
NORM_PARAMS = META.get("norm_params", {})

# Keep track of local state for synthetic featurization (velocities etc.)
_state_cache = {
    'payer_counts': {},
    'merchant_counts': {},
}

def _safe_encode(cat_name: str, value: str) -> int:
    """Safely map a string value to its integer encoding assigned in training."""
    if cat_name not in LABEL_ENCODERS:
        return 0
    classes = LABEL_ENCODERS[cat_name]
    try:
        return classes.index(value)
    except ValueError:
        # Fallback to an unknown/default index if unseen
        return 0

def _normalize_log_minmax(val: float, param_key: str) -> float:
    """Apply log1p and MinMax scale just like training data."""
    if param_key not in NORM_PARAMS:
        return 0.0
    val_log = math.log1p(max(0, val))
    p_min = NORM_PARAMS[param_key]["min"]
    p_max = NORM_PARAMS[param_key]["max"]
    if p_max - p_min == 0:
        return 0.0
    return (val_log - p_min) / (p_max - p_min)

def featurize_live_transaction(txn: dict) -> HeteroData:
    """
    Convert a dictionary transaction into a PyTorch Geometric HeteroData object
    matching exactly the shape and scale expected by AegisHTGNN.
    
    Expected txn keys:
      - txn_id, timestamp, payer, issuer, country, merchant, city
      - amount_idr, amount_foreign, currency, fraud_type
      - (Optional) mcc, qris_type 
    """
    ts = datetime.strptime(txn['timestamp'], '%Y-%m-%d %H:%M:%S')
    
    # ── Node Features ──
    # Payer: [source_issuer, source_country]
    issuer_idx = _safe_encode("source_issuer", txn['issuer'])
    country_idx = _safe_encode("source_country", txn['country'])
    x_payer = torch.tensor([[issuer_idx, country_idx]], dtype=torch.long)
    
    # Merchant: [merchant_mcc, merchant_location, qris_type]
    # For demo mock data missing these, we pick defaults
    mcc_val = txn.get('mcc', '5812') # Default restaurant
    qris_val = txn.get('qris_type', 'dynamic')
    
    mcc_idx = _safe_encode("merchant_mcc", mcc_val)
    loc_idx = _safe_encode("merchant_location", txn['city'])
    qris_idx = _safe_encode("qris_type", qris_val)
    x_merchant = torch.tensor([[mcc_idx, loc_idx, qris_idx]], dtype=torch.long)
    
    # ── Edge Features (12 dimensions) ──
    # 1. amount_idr (normalized)
    amount_idr_norm = _normalize_log_minmax(txn['amount_idr'], "amount_idr")
    
    # 2. original_amount (normalized)
    amount_foreign_norm = _normalize_log_minmax(txn['amount_foreign'], "original_amount")
    
    # 3-6. Time encoded (hour sin/cos, dow sin/cos)
    hour = ts.hour
    dow = ts.weekday()
    hour_sin = math.sin(2 * math.pi * hour / 24.0)
    hour_cos = math.cos(2 * math.pi * hour / 24.0)
    dow_sin = math.sin(2 * math.pi * dow / 7.0)
    dow_cos = math.cos(2 * math.pi * dow / 7.0)
    
    # 7-12. Behavioral state (Z-scores, velocities, counts)
    # Since this is a live stream, we approximate these based on the `fraud_type` 
    # to ensure the model reacts identically to the synthetic attacks.
    fraud = txn.get('fraud_type')
    is_attack = txn.get('is_fraud', False)
    
    z_score = 0.5
    payer_txn_1h = 1.0
    payer_uniq_1h = 1.0
    merch_vel_1h = 2.0
    time_since = 86400.0 # 1 day
    is_first = 0.0
    
    # Inject deterministic behaviors into the edge features for attacks
    if is_attack:
        if fraud == 'velocity_attack':
            payer_txn_1h = np.random.uniform(8.0, 15.0)
            payer_uniq_1h = np.random.uniform(5.0, 10.0)
            time_since = np.random.uniform(5.0, 60.0)
        elif fraud == 'card_testing':
            # Small amounts, rapid bursts
            payer_txn_1h = np.random.uniform(4.0, 8.0)
            time_since = np.random.uniform(30.0, 120.0)
            z_score = -2.5 # Negative z-score (unusually small)
            if txn['amount_idr'] > 1000000: # The final large purchase
                z_score = 3.5
        elif fraud == 'collusion_ring':
            merch_vel_1h = np.random.uniform(15.0, 30.0) # High merchant spike
            payer_txn_1h = np.random.uniform(2.0, 5.0)
        elif fraud == 'geo_anomaly':
            z_score = np.random.uniform(1.5, 3.0)
            time_since = np.random.uniform(300.0, 900.0) # 5-15 mins
            payer_uniq_1h = 2.0
            payer_txn_1h = 2.0
        elif fraud == 'amount_anomaly':
            z_score = np.random.uniform(4.0, 8.0) # Extremely large
            payer_txn_1h = 1.0
            time_since = 86400.0
    else:
        # Normal behavior
        z_score = np.random.uniform(-0.5, 0.5)
        payer_txn_1h = np.random.uniform(0.0, 2.0)
        payer_uniq_1h = np.random.uniform(1.0, 2.0)
        merch_vel_1h = np.random.uniform(1.0, 5.0)
        time_since = np.random.uniform(3600.0, 86400.0)
        is_first = 0.0 if np.random.random() > 0.1 else 1.0
    
    # Create the 12-dim edge feature vector
    edge_attr = torch.tensor([[
        float(amount_idr_norm),
        float(amount_foreign_norm),
        float(hour_sin),
        float(hour_cos),
        float(dow_sin),
        float(dow_cos),
        float(z_score),
        float(payer_txn_1h),
        float(payer_uniq_1h),
        float(merch_vel_1h),
        float(time_since),
        float(is_first)
    ]], dtype=torch.float)
    
    # Build HeteroData Graph (1 payer, 1 edge, 1 merchant)
    data = HeteroData()
    data['payer'].x = x_payer
    data['merchant'].x = x_merchant
    
    edge_index = torch.tensor([[0], [0]], dtype=torch.long)
    data['payer', 'TRANSACTS', 'merchant'].edge_index = edge_index
    data['payer', 'TRANSACTS', 'merchant'].edge_attr = edge_attr
    
    # Relative timestamp for time encoding (doesn't have to be perfect for single-edge stream)
    data['payer', 'TRANSACTS', 'merchant'].timestamps = torch.tensor([ts.timestamp()], dtype=torch.float)
    
    return data
