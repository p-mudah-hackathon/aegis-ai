import numpy as np
import hashlib

# ============================================================
# RANDOM SEED & DATE RANGE
# ============================================================
SEED = 42

# ============================================================
# ISSUER DEFINITIONS
# ============================================================
ISSUERS = [
    {'name': 'Alipay',      'country': 'CN', 'currency': 'CNY', 'rate': 2190,  'w': 0.30},
    {'name': 'WeChat_Pay',   'country': 'CN', 'currency': 'CNY', 'rate': 2190,  'w': 0.18},
    {'name': 'Kakaopay',     'country': 'KR', 'currency': 'KRW', 'rate': 10.8,  'w': 0.13},
    {'name': 'LINE_Pay',     'country': 'JP', 'currency': 'JPY', 'rate': 96,    'w': 0.11},
    {'name': 'PayPay',       'country': 'JP', 'currency': 'JPY', 'rate': 96,    'w': 0.06},
    {'name': 'Touch_n_Go',   'country': 'MY', 'currency': 'MYR', 'rate': 3450,  'w': 0.08},
    {'name': 'GrabPay_SG',   'country': 'SG', 'currency': 'SGD', 'rate': 11600, 'w': 0.07},
    {'name': 'PromptPay',    'country': 'TH', 'currency': 'THB', 'rate': 440,   'w': 0.04},
    {'name': 'UnionPay',     'country': 'CN', 'currency': 'CNY', 'rate': 2190,  'w': 0.03},
]

# ============================================================
# MERCHANT CATEGORY CODES
# ============================================================
MCC = {
    '5812': {'name': 'Restaurants',     'avg': 180000,  'std': 95000,   'w': 0.25},
    '5411': {'name': 'Grocery',         'avg': 150000,  'std': 80000,   'w': 0.10},
    '5399': {'name': 'General_Retail',  'avg': 350000,  'std': 200000,  'w': 0.15},
    '7011': {'name': 'Hotels',          'avg': 1500000, 'std': 900000,  'w': 0.08},
    '5691': {'name': 'Clothing',        'avg': 400000,  'std': 220000,  'w': 0.10},
    '7832': {'name': 'Entertainment',   'avg': 250000,  'std': 150000,  'w': 0.07},
    '5541': {'name': 'Gas_Stations',    'avg': 200000,  'std': 100000,  'w': 0.04},
    '7512': {'name': 'Car_Rental',      'avg': 850000,  'std': 450000,  'w': 0.03},
    '5944': {'name': 'Jewelry',         'avg': 2800000, 'std': 1800000, 'w': 0.04},
    '5999': {'name': 'Specialty',       'avg': 300000,  'std': 170000,  'w': 0.06},
    '5947': {'name': 'Gift_Souvenir',   'avg': 200000,  'std': 120000,  'w': 0.05},
    '5814': {'name': 'Fast_Food',       'avg': 80000,   'std': 40000,   'w': 0.03},
}

# ============================================================
# LOCATION DEFINITIONS
# ============================================================
LOCATIONS = [
    {'city': 'Bali',        'w': 0.28, 'lat': -8.41,  'lon': 115.19},
    {'city': 'Jakarta',     'w': 0.22, 'lat': -6.21,  'lon': 106.85},
    {'city': 'Yogyakarta',  'w': 0.10, 'lat': -7.80,  'lon': 110.37},
    {'city': 'Bandung',     'w': 0.08, 'lat': -6.92,  'lon': 107.62},
    {'city': 'Surabaya',    'w': 0.07, 'lat': -7.26,  'lon': 112.75},
    {'city': 'Lombok',      'w': 0.06, 'lat': -8.65,  'lon': 116.32},
    {'city': 'Medan',       'w': 0.05, 'lat': 3.60,   'lon': 98.67},
    {'city': 'Makassar',    'w': 0.05, 'lat': -5.15,  'lon': 119.43},
    {'city': 'Semarang',    'w': 0.05, 'lat': -6.97,  'lon': 110.42},
    {'city': 'Malang',      'w': 0.04, 'lat': -7.97,  'lon': 112.63},
]

# ============================================================
# TEMPORAL WEIGHTS
# ============================================================
# Hour-of-day probability weights (tourist spending patterns)
HOUR_WEIGHTS = np.array([
    0.005, 0.003, 0.002, 0.002, 0.003, 0.005,  # 0-5   (very low)
    0.01,  0.02,  0.04,  0.06,                  # 6-9   (morning)
    0.08,  0.09,  0.09,  0.07,                  # 10-13 (peak 1)
    0.05,  0.04,  0.05,  0.06,                  # 14-17 (afternoon)
    0.08,  0.09,  0.08,  0.06,                  # 18-21 (peak 2)
    0.04,  0.02,                                 # 22-23
])
HOUR_WEIGHTS /= HOUR_WEIGHTS.sum()


# ============================================================
# UTILITY FUNCTIONS
# ============================================================
def haversine_km(lat1, lon1, lat2, lon2):
    # Calculate distance in km between two lat/lon points.
    R = 6371
    dlat, dlon = np.radians(lat2 - lat1), np.radians(lon2 - lon1)
    a = np.sin(dlat/2)**2 + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(dlon/2)**2
    return 2 * R * np.arcsin(np.sqrt(a))


def hash_token(raw_id):
    # Hash a raw ID to a 16-char hex token.
    return hashlib.sha256(raw_id.encode()).hexdigest()[:16]
