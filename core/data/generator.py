import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import json
import os
import bisect
import warnings
warnings.filterwarnings('ignore')

from core.data.constants import (
    SEED, ISSUERS, MCC, LOCATIONS, HOUR_WEIGHTS,
    haversine_km, hash_token,
)

# Date range for generation
DATE_START = datetime(2026, 1, 1)
DATE_END = datetime(2026, 2, 28)


# ============================================================
# GENERATOR
# ============================================================
class AegisDataGenerator:
    def __init__(self, n_payers=5000, n_merchants=500):
        self.rng = np.random.default_rng(SEED)
        self.n_payers = n_payers
        self.n_merchants = n_merchants
        self.issuer_names = [i['name'] for i in ISSUERS]
        self.issuer_weights = np.array([i['w'] for i in ISSUERS])
        self.issuer_weights /= self.issuer_weights.sum()
        self.issuer_map = {i['name']: i for i in ISSUERS}
        self.mcc_codes = list(MCC.keys())
        self.mcc_weights = np.array([MCC[c]['w'] for c in self.mcc_codes])
        self.mcc_weights /= self.mcc_weights.sum()
        self.loc_cities = [l['city'] for l in LOCATIONS]
        self.loc_weights = np.array([l['w'] for l in LOCATIONS])
        self.loc_weights /= self.loc_weights.sum()
        self.loc_map = {l['city']: l for l in LOCATIONS}
        self.scenario_id = 0

    def _next_scenario(self):
        self.scenario_id += 1
        return f"FS_{self.scenario_id:04d}"

    # ------ Entity Creation ------
    def _create_merchants(self):
        merchants = []
        for i in range(self.n_merchants):
            city = self.rng.choice(self.loc_cities, p=self.loc_weights)
            mcc = self.rng.choice(self.mcc_codes, p=self.mcc_weights)
            qris_type = 'static' if self.rng.random() < 0.35 else 'dynamic'
            # Each merchant has slight amount variation from MCC average
            base = MCC[mcc]
            m_avg = base['avg'] * self.rng.uniform(0.7, 1.3)
            m_std = base['std'] * self.rng.uniform(0.6, 1.4)
            merchants.append({
                'nmid': f"NM{city[:3].upper()}{i:04d}",
                'mcc': mcc,
                'city': city,
                'qris_type': qris_type,
                'avg_amount': m_avg,
                'std_amount': m_std,
            })
        return pd.DataFrame(merchants)

    def _create_payers(self):
        payers = []
        total_days = (DATE_END - DATE_START).days
        for i in range(self.n_payers):
            issuer_name = self.rng.choice(self.issuer_names, p=self.issuer_weights)
            issuer = self.issuer_map[issuer_name]
            trip_len = int(self.rng.integers(3, 8))
            max_start = total_days - trip_len
            start_off = int(self.rng.integers(0, max(1, max_start)))
            trip_start = DATE_START + timedelta(days=start_off)
            trip_end = trip_start + timedelta(days=trip_len)
            base_city = self.rng.choice(self.loc_cities, p=self.loc_weights)
            budget = self.rng.choice(['budget', 'mid', 'luxury'], p=[0.40, 0.45, 0.15])
            daily_mean = {'budget': 2.5, 'mid': 3.5, 'luxury': 4.5}[budget]
            amount_mult = {'budget': 0.6, 'mid': 1.0, 'luxury': 2.0}[budget]
            payers.append({
                'token': hash_token(f"PTK_{i:05d}"),
                'issuer': issuer_name,
                'country': issuer['country'],
                'currency': issuer['currency'],
                'rate': issuer['rate'],
                'trip_start': trip_start,
                'trip_end': trip_end,
                'base_city': base_city,
                'daily_mean': daily_mean,
                'amount_mult': amount_mult,
            })
        return pd.DataFrame(payers)

    # ------ Amount Generation ------
    def _gen_amount(self, mcc_code, amount_mult=1.0, merchant=None):
        if merchant is not None:
            mu, sig = merchant['avg_amount'], merchant['std_amount']
        else:
            base = MCC[mcc_code]
            mu, sig = base['avg'], base['std']
        # Log-normal for realistic financial distribution
        log_mu = np.log(mu) - 0.5 * np.log(1 + (sig/mu)**2)
        log_sig = np.sqrt(np.log(1 + (sig/mu)**2))
        amt = self.rng.lognormal(log_mu, log_sig) * amount_mult
        amt = max(10000, int(round(amt / 1000) * 1000))  # Round to nearest 1000, min 10K
        amt = min(amt, 10000000)  # max 10,000,000 IDR
        return amt

    def _sample_hour(self):
        return int(self.rng.choice(24, p=HOUR_WEIGHTS))

    # ------ Normal Transaction Generation ------
    def _generate_normal(self, merchants_df, payers_df):
        print("  Generating normal transactions...")
        txns = []
        m_by_city = merchants_df.groupby('city')

        for _, p in payers_df.iterrows():
            cur = p['trip_start']
            # Get merchants in payer's base city (80%) + nearby (20%)
            city_merchants = merchants_df[merchants_df['city'] == p['base_city']]
            other_merchants = merchants_df[merchants_df['city'] != p['base_city']]

            while cur <= p['trip_end']:
                n_txns = max(0, int(self.rng.poisson(p['daily_mean'])))
                for _ in range(n_txns):
                    # 80% in base city, 20% elsewhere
                    if self.rng.random() < 0.80 and len(city_merchants) > 0:
                        m = city_merchants.iloc[self.rng.integers(len(city_merchants))]
                    else:
                        m = other_merchants.iloc[self.rng.integers(len(other_merchants))]

                    h = self._sample_hour()
                    ts = cur.replace(hour=h, minute=int(self.rng.integers(60)),
                                     second=int(self.rng.integers(60)))
                    amt = self._gen_amount(m['mcc'], p['amount_mult'], m)
                    orig = round(amt / p['rate'], 2)

                    txns.append({
                        'timestamp': ts,
                        'payer_token_id': p['token'],
                        'source_issuer': p['issuer'],
                        'source_country': p['country'],
                        'merchant_nmid': m['nmid'],
                        'merchant_mcc': m['mcc'],
                        'merchant_location': m['city'],
                        'merchant_country': 'ID',
                        'qris_type': m['qris_type'],
                        'amount_idr': amt,
                        'original_amount': orig,
                        'original_currency': p['currency'],
                        'is_cross_border': 1,
                        'is_fraud': 0,
                        'fraud_type': 'legitimate',
                        'fraud_scenario_id': '',
                    })
                cur += timedelta(days=1)

        print(f"    -> {len(txns)} normal transactions")
        return txns

    # ------ Fraud Injection ------
    def _inject_velocity(self, payers_df, merchants_df, n_scenarios=150):
        # Rapid transactions across multiple merchants in <20 minutes.
        print(f"  Injecting velocity_attack ({n_scenarios} scenarios)...")
        txns = []
        for _ in range(n_scenarios):
            sid = self._next_scenario()
            p = payers_df.iloc[self.rng.integers(len(payers_df))]
            n_tx = int(self.rng.integers(6, 14))
            # Start at off-hours
            base_day = p['trip_start'] + timedelta(days=int(self.rng.integers(0,
                max(1, (p['trip_end'] - p['trip_start']).days))))
            start_hour = self.rng.choice([0, 1, 2, 3, 23])
            base_ts = base_day.replace(hour=start_hour, minute=int(self.rng.integers(60)))

            city_m = merchants_df[merchants_df['city'] == p['base_city']]
            if len(city_m) < 3:
                city_m = merchants_df
            selected = city_m.iloc[self.rng.choice(len(city_m), size=min(n_tx, len(city_m)), replace=False)]

            for j in range(n_tx):
                offset = int(self.rng.integers(30, 150))  # 30s-2.5min between txns
                ts = base_ts + timedelta(seconds=j * offset)
                m = selected.iloc[j % len(selected)]
                amt = self._gen_amount(m['mcc'], p['amount_mult'] * 1.5, m)
                txns.append({
                    'timestamp': ts,
                    'payer_token_id': p['token'],
                    'source_issuer': p['issuer'],
                    'source_country': p['country'],
                    'merchant_nmid': m['nmid'],
                    'merchant_mcc': m['mcc'],
                    'merchant_location': m['city'],
                    'merchant_country': 'ID',
                    'qris_type': m['qris_type'],
                    'amount_idr': amt,
                    'original_amount': round(amt / p['rate'], 2),
                    'original_currency': p['currency'],
                    'is_cross_border': 1,
                    'is_fraud': 1,
                    'fraud_type': 'velocity_attack',
                    'fraud_scenario_id': sid,
                })
        print(f"    -> {len(txns)} fraud txns")
        return txns

    def _inject_card_testing(self, merchants_df, n_scenarios=80):
        # Small amounts at many merchants, then one large purchase.
        print(f"  Injecting card_testing ({n_scenarios} scenarios)...")
        txns = []
        for _ in range(n_scenarios):
            sid = self._next_scenario()
            issuer_name = self.rng.choice(self.issuer_names, p=self.issuer_weights)
            issuer = self.issuer_map[issuer_name]
            token = hash_token(f"FRAUD_CT_{self.scenario_id}")
            city = self.rng.choice(self.loc_cities, p=self.loc_weights)
            city_m = merchants_df[merchants_df['city'] == city]
            if len(city_m) < 3:
                city_m = merchants_df

            day_offset = int(self.rng.integers(0, (DATE_END - DATE_START).days))
            base_ts = DATE_START + timedelta(days=day_offset,
                hours=int(self.rng.integers(8, 22)), minutes=int(self.rng.integers(60)))

            n_probes = int(self.rng.integers(4, 9))
            selected = city_m.iloc[self.rng.choice(len(city_m),
                size=min(n_probes + 1, len(city_m)), replace=False)]

            # Small probe transactions
            for j in range(n_probes):
                ts = base_ts + timedelta(seconds=int(self.rng.integers(60, 300)) * j)
                m = selected.iloc[j % len(selected)]
                amt = int(self.rng.integers(10, 35)) * 1000  # 10K-35K IDR
                txns.append({
                    'timestamp': ts,
                    'payer_token_id': token,
                    'source_issuer': issuer_name,
                    'source_country': issuer['country'],
                    'merchant_nmid': m['nmid'],
                    'merchant_mcc': m['mcc'],
                    'merchant_location': m['city'],
                    'merchant_country': 'ID',
                    'qris_type': m['qris_type'],
                    'amount_idr': amt,
                    'original_amount': round(amt / issuer['rate'], 2),
                    'original_currency': issuer['currency'],
                    'is_cross_border': 1,
                    'is_fraud': 1,
                    'fraud_type': 'card_testing',
                    'fraud_scenario_id': sid,
                })

            # Final large purchase
            ts_big = ts + timedelta(seconds=int(self.rng.integers(120, 600)))
            m_big = selected.iloc[-1]
            amt_big = int(self.rng.integers(3, 10)) * 1000000  # 3M-10M IDR (respects QRIS limit)
            txns.append({
                'timestamp': ts_big,
                'payer_token_id': token,
                'source_issuer': issuer_name,
                'source_country': issuer['country'],
                'merchant_nmid': m_big['nmid'],
                'merchant_mcc': m_big['mcc'],
                'merchant_location': m_big['city'],
                'merchant_country': 'ID',
                'qris_type': m_big['qris_type'],
                'amount_idr': amt_big,
                'original_amount': round(amt_big / issuer['rate'], 2),
                'original_currency': issuer['currency'],
                'is_cross_border': 1,
                'is_fraud': 1,
                'fraud_type': 'card_testing',
                'fraud_scenario_id': sid,
            })
        print(f"    -> {len(txns)} fraud txns")
        return txns

    def _inject_collusion_ring(self, merchants_df, n_scenarios=35):
        # Multiple coordinated payers hitting same merchant(s).
        print(f"  Injecting collusion_ring ({n_scenarios} scenarios)...")
        txns = []
        for _ in range(n_scenarios):
            sid = self._next_scenario()
            n_members = int(self.rng.integers(3, 7))
            issuer_name = self.rng.choice(self.issuer_names, p=self.issuer_weights)
            issuer = self.issuer_map[issuer_name]

            # All members hit 1-2 target merchants
            target_m = merchants_df.iloc[self.rng.choice(len(merchants_df),
                size=self.rng.integers(1, 3), replace=False)]

            day_offset = int(self.rng.integers(0, (DATE_END - DATE_START).days))
            base_ts = DATE_START + timedelta(days=day_offset,
                hours=int(self.rng.integers(10, 20)))

            base_amount = int(self.rng.integers(500, 5000)) * 1000  # 500K-5M

            for mi in range(n_members):
                token = hash_token(f"FRAUD_CR_{self.scenario_id}_{mi}")
                for _, m in target_m.iterrows():
                    for rep in range(self.rng.integers(1, 4)):
                        offset_min = int(self.rng.integers(0, 120))
                        ts = base_ts + timedelta(minutes=offset_min + mi * 5)
                        amt = int(base_amount * self.rng.uniform(0.8, 1.2))
                        txns.append({
                            'timestamp': ts,
                            'payer_token_id': token,
                            'source_issuer': issuer_name,
                            'source_country': issuer['country'],
                            'merchant_nmid': m['nmid'],
                            'merchant_mcc': m['mcc'],
                            'merchant_location': m['city'],
                            'merchant_country': 'ID',
                            'qris_type': m['qris_type'],
                            'amount_idr': amt,
                            'original_amount': round(amt / issuer['rate'], 2),
                            'original_currency': issuer['currency'],
                            'is_cross_border': 1,
                            'is_fraud': 1,
                            'fraud_type': 'collusion_ring',
                            'fraud_scenario_id': sid,
                        })
        print(f"    -> {len(txns)} fraud txns")
        return txns

    def _inject_geo_anomaly(self, payers_df, merchants_df, n_scenarios=100):
        # Same payer in distant cities within impossible timeframe.
        print(f"  Injecting geo_anomaly ({n_scenarios} scenarios)...")
        txns = []
        # Find pairs of distant cities
        distant_pairs = []
        for i, l1 in enumerate(LOCATIONS):
            for j, l2 in enumerate(LOCATIONS):
                if i < j:
                    d = haversine_km(l1['lat'], l1['lon'], l2['lat'], l2['lon'])
                    if d > 300:
                        distant_pairs.append((l1['city'], l2['city'], d))

        for _ in range(n_scenarios):
            sid = self._next_scenario()
            p = payers_df.iloc[self.rng.integers(len(payers_df))]
            city1, city2, dist = distant_pairs[self.rng.integers(len(distant_pairs))]

            day_offset = int(self.rng.integers(0,
                max(1, (p['trip_end'] - p['trip_start']).days)))
            base_day = p['trip_start'] + timedelta(days=day_offset)
            h = int(self.rng.integers(9, 21))
            ts1 = base_day.replace(hour=h, minute=int(self.rng.integers(60)))
            ts2 = ts1 + timedelta(minutes=int(self.rng.integers(5, 25)))

            m1_pool = merchants_df[merchants_df['city'] == city1]
            m2_pool = merchants_df[merchants_df['city'] == city2]
            if len(m1_pool) == 0 or len(m2_pool) == 0:
                continue
            m1 = m1_pool.iloc[self.rng.integers(len(m1_pool))]
            m2 = m2_pool.iloc[self.rng.integers(len(m2_pool))]

            for ts, m in [(ts1, m1), (ts2, m2)]:
                amt = self._gen_amount(m['mcc'], p['amount_mult'], m)
                txns.append({
                    'timestamp': ts,
                    'payer_token_id': p['token'],
                    'source_issuer': p['issuer'],
                    'source_country': p['country'],
                    'merchant_nmid': m['nmid'],
                    'merchant_mcc': m['mcc'],
                    'merchant_location': m['city'],
                    'merchant_country': 'ID',
                    'qris_type': m['qris_type'],
                    'amount_idr': amt,
                    'original_amount': round(amt / p['rate'], 2),
                    'original_currency': p['currency'],
                    'is_cross_border': 1,
                    'is_fraud': 1,
                    'fraud_type': 'geo_anomaly',
                    'fraud_scenario_id': sid,
                })
        print(f"    -> {len(txns)} fraud txns")
        return txns

    def _inject_amount_anomaly(self, payers_df, merchants_df, n_scenarios=200):
        # Extreme amounts at off-hours.
        print(f"  Injecting amount_anomaly ({n_scenarios} scenarios)...")
        txns = []
        for _ in range(n_scenarios):
            sid = self._next_scenario()
            p = payers_df.iloc[self.rng.integers(len(payers_df))]
            m = merchants_df.iloc[self.rng.integers(len(merchants_df))]

            day_offset = int(self.rng.integers(0,
                max(1, (p['trip_end'] - p['trip_start']).days)))
            base_day = p['trip_start'] + timedelta(days=day_offset)
            h = self.rng.choice([0, 1, 2, 3, 4, 22, 23])
            ts = base_day.replace(hour=h, minute=int(self.rng.integers(60)))

            # 5-15x the merchant average, but capped at 10,000,000 IDR limit
            amt = int(m['avg_amount'] * self.rng.uniform(5, 15))
            amt = int(round(amt / 1000) * 1000)
            amt = min(amt, 10000000)
            # Ensure it is at least noticeably large if the merchant average is extremely low
            amt = max(amt, 3000000)
            # Final capping just to be absolutely safe
            amt = min(amt, 10000000)

            txns.append({
                'timestamp': ts,
                'payer_token_id': p['token'],
                'source_issuer': p['issuer'],
                'source_country': p['country'],
                'merchant_nmid': m['nmid'],
                'merchant_mcc': m['mcc'],
                'merchant_location': m['city'],
                'merchant_country': 'ID',
                'qris_type': m['qris_type'],
                'amount_idr': amt,
                'original_amount': round(amt / p['rate'], 2),
                'original_currency': p['currency'],
                'is_cross_border': 1,
                'is_fraud': 1,
                'fraud_type': 'amount_anomaly',
                'fraud_scenario_id': sid,
            })
        print(f"    -> {len(txns)} fraud txns")
        return txns

    # ------ Feature Engineering ------
    def _compute_features(self, df):
        print("  Computing engineered features...")
        df = df.sort_values('timestamp').reset_index(drop=True)
        df['txn_id'] = [f"TX-{i+1:06d}" for i in range(len(df))]
        df['hour_of_day'] = df['timestamp'].dt.hour
        df['day_of_week'] = df['timestamp'].dt.dayofweek

        # -- Per-payer features --
        df['time_since_last_txn_sec'] = -1.0
        df['payer_txn_count_1h'] = 0
        df['payer_unique_merchant_1h'] = 0
        df['is_first_txn_payer'] = 0

        ts_epoch = df['timestamp'].astype(np.int64) // 10**9

        print("    Per-payer rolling features...")
        for token, grp in df.groupby('payer_token_id'):
            idx = grp.index.values
            ts = ts_epoch.values[idx]
            merchants = df['merchant_nmid'].values[idx]
            n = len(idx)

            tslast = np.full(n, -1.0)
            cnt1h = np.zeros(n, dtype=int)
            uniq1h = np.zeros(n, dtype=int)
            first = np.zeros(n, dtype=int)
            first[0] = 1

            for i in range(1, n):
                tslast[i] = ts[i] - ts[i-1]
                # Count in 1-hour window using binary search
                threshold = ts[i] - 3600
                left = bisect.bisect_left(ts, threshold, 0, i)
                cnt1h[i] = i - left
                uniq1h[i] = len(set(merchants[left:i]))

            df.loc[idx, 'time_since_last_txn_sec'] = tslast
            df.loc[idx, 'payer_txn_count_1h'] = cnt1h
            df.loc[idx, 'payer_unique_merchant_1h'] = uniq1h
            df.loc[idx, 'is_first_txn_payer'] = first

        # -- Per-merchant features --
        print("    Per-merchant velocity...")
        merch_vel = np.zeros(len(df), dtype=int)
        for nmid, grp in df.groupby('merchant_nmid'):
            idx = grp.index.values
            ts = ts_epoch.values[idx]
            for i in range(len(idx)):
                threshold = ts[i] - 3600
                left = bisect.bisect_left(ts, threshold, 0, i)
                merch_vel[idx[i]] = i - left
        df['merchant_txn_velocity_1h'] = merch_vel

        # -- Amount z-score per merchant --
        print("    Amount z-scores...")
        m_stats = df.groupby('merchant_nmid')['amount_idr'].agg(['mean', 'std'])
        m_stats.columns = ['_m_avg', '_m_std']
        m_stats['_m_std'] = m_stats['_m_std'].replace(0, 1)
        df = df.merge(m_stats, left_on='merchant_nmid', right_index=True, how='left')
        df['amount_zscore_merchant'] = ((df['amount_idr'] - df['_m_avg']) / df['_m_std']).round(3)
        df.drop(columns=['_m_avg', '_m_std'], inplace=True)

        return df

    # ------ Main Pipeline ------
    def generate(self):
        print("=" * 60)
        print("AegisNode Synthetic Dataset Generator")
        print("=" * 60)

        print("\n[1/4] Creating entities...")
        merchants = self._create_merchants()
        payers = self._create_payers()
        print(f"  {len(merchants)} merchants, {len(payers)} payers")

        print("\n[2/4] Generating transactions...")
        all_txns = self._generate_normal(merchants, payers)

        print("\n[3/4] Injecting fraud scenarios...")
        all_txns += self._inject_velocity(payers, merchants, n_scenarios=90)
        all_txns += self._inject_card_testing(merchants, n_scenarios=85)
        all_txns += self._inject_collusion_ring(merchants, n_scenarios=45)
        all_txns += self._inject_geo_anomaly(payers, merchants, n_scenarios=225)
        all_txns += self._inject_amount_anomaly(payers, merchants, n_scenarios=450)

        df = pd.DataFrame(all_txns)
        print(f"\n  Total raw transactions: {len(df)}")

        print("\n[4/4] Feature engineering...")
        df = self._compute_features(df)

        # Reorder columns
        col_order = [
            'txn_id', 'timestamp', 'payer_token_id', 'source_issuer', 'source_country',
            'merchant_nmid', 'merchant_mcc', 'merchant_location', 'merchant_country',
            'qris_type', 'amount_idr', 'original_amount', 'original_currency',
            'is_cross_border', 'hour_of_day', 'day_of_week',
            'time_since_last_txn_sec', 'payer_txn_count_1h', 'payer_unique_merchant_1h',
            'merchant_txn_velocity_1h', 'amount_zscore_merchant', 'is_first_txn_payer',
            'is_fraud', 'fraud_type', 'fraud_scenario_id',
        ]
        df = df[col_order]
        return df

    def save(self, df, outdir):
        # Save CSV
        csv_path = os.path.join(outdir, 'aegis_qris_transactions.csv')
        df.to_csv(csv_path, index=False)
        print(f"\n  Saved: {csv_path} ({len(df)} rows)")

        # Save graph metadata
        meta = {
            "graph_definition": {
                "node_types": {
                    "payer": {
                        "id_column": "payer_token_id",
                        "static_features": ["source_issuer", "source_country"],
                        "dynamic_features": ["payer_txn_count_1h", "payer_unique_merchant_1h",
                                             "time_since_last_txn_sec", "is_first_txn_payer"],
                    },
                    "merchant": {
                        "id_column": "merchant_nmid",
                        "static_features": ["merchant_mcc", "merchant_location",
                                            "merchant_country", "qris_type"],
                        "dynamic_features": ["merchant_txn_velocity_1h"],
                    },
                },
                "edge_types": {
                    "TRANSACTS": {
                        "source_node": "payer",
                        "target_node": "merchant",
                        "temporal": True,
                        "features": ["amount_idr", "original_amount", "hour_of_day",
                                     "day_of_week", "amount_zscore_merchant"],
                        "label": "is_fraud",
                    }
                },
            },
            "fraud_types": {
                "velocity_attack": "Rapid transactions (6-14) across multiple merchants in <20 min, typically off-hours",
                "card_testing": "Multiple small probing amounts (10K-35K) at different merchants, followed by large purchase",
                "collusion_ring": "3-7 coordinated payer tokens from same issuer hitting 1-2 merchants within 2 hours",
                "geo_anomaly": "Same payer transacting in cities >300km apart within 5-25 minutes",
                "amount_anomaly": "Transaction amount 5-15x merchant average at off-hours (10PM-5AM)",
            },
            "dataset_stats": {
                "total_transactions": int(len(df)),
                "total_fraud": int(df['is_fraud'].sum()),
                "fraud_rate": round(df['is_fraud'].mean() * 100, 2),
                "unique_payers": int(df['payer_token_id'].nunique()),
                "unique_merchants": int(df['merchant_nmid'].nunique()),
                "date_range": f"{df['timestamp'].min()} to {df['timestamp'].max()}",
                "fraud_breakdown": df[df['is_fraud']==1]['fraud_type'].value_counts().to_dict(),
            },
        }
        meta_path = os.path.join(outdir, 'aegis_graph_meta.json')
        with open(meta_path, 'w') as f:
            json.dump(meta, f, indent=2, default=str)
        print(f"  Saved: {meta_path}")

        # Print summary
        print("\n" + "=" * 60)
        print("DATASET SUMMARY")
        print("=" * 60)
        print(f"  Total transactions : {len(df):,}")
        print(f"  Fraud transactions : {df['is_fraud'].sum():,} ({df['is_fraud'].mean()*100:.2f}%)")
        print(f"  Unique payers      : {df['payer_token_id'].nunique():,}")
        print(f"  Unique merchants   : {df['merchant_nmid'].nunique():,}")
        print(f"  Date range         : {df['timestamp'].min()} to {df['timestamp'].max()}")
        print(f"\n  Fraud breakdown:")
        for ft, cnt in df[df['is_fraud']==1]['fraud_type'].value_counts().items():
            print(f"    {ft:20s}: {cnt:,}")
        print(f"\n  Amount (IDR) stats:")
        print(f"    Mean   : {df['amount_idr'].mean():,.0f}")
        print(f"    Median : {df['amount_idr'].median():,.0f}")
        print(f"    Std    : {df['amount_idr'].std():,.0f}")
        print(f"    Min    : {df['amount_idr'].min():,.0f}")
        print(f"    Max    : {df['amount_idr'].max():,.0f}")
        print(f"\n  Issuer distribution:")
        for iss, cnt in df['source_issuer'].value_counts().items():
            print(f"    {iss:15s}: {cnt:,} ({cnt/len(df)*100:.1f}%)")
        print("=" * 60)
