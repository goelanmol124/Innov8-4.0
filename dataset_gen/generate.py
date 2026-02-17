"""
Synthetic fraud detection dataset generator.

Usage:
    # Generate participant dataset (seed 42)
    python generate.py --split participant --seed 42 --output-dir ../participant_kit

    # Generate evaluation dataset (seed 99, KEEP SECRET)
    python generate.py --split eval --seed 99 --output-dir ../eval_dataset

Same function + different seed = structurally identical but statistically
different dataset. Participants never see the eval seed or eval dataset.

DO NOT share this file with participants.

Design targets (verified after generation):
    - No single feature has AUC > 0.72 against the fraud label
    - Random 100-sample baseline (logistic regression): F1 ≈ 0.30 – 0.45
    - Cluster-aware sampling (50+ fraud in 100 queries): F1 ≈ 0.65 – 0.78
    - Expert active learning + strong classifier: F1 ≈ 0.85+
"""

import numpy as np
import pandas as pd
from pathlib import Path

# ─── Feature column order (must stay stable across seeds) ─────────────────────
FEATURES = [
    "amount",
    "amount_log",
    "hour",
    "day_of_week",
    "user_account_age_days",
    "card_age_days",
    "txn_count_7d",
    "txn_count_30d",
    "avg_amount_30d",
    "std_amount_30d",
    "amount_to_avg_ratio",
    "failed_auths_24h",
    "is_new_device",
    "is_international",
    "distance_km",
    "ip_risk_score",
    "email_risk_score",
    "merchant_risk_category",
    "time_since_last_txn_hrs",
    "device_os_encoded",
    "browser_encoded",
    "feature_noise_1",
    "feature_noise_2",
    "feature_noise_3",
    "feature_noise_4",
]


# ─── Legitimate transaction generator ─────────────────────────────────────────

def _gen_legit(rng, n):
    """
    Legitimate transactions. Distributions are intentionally wider / noisier
    than a typical "clean" dataset so that fraud clusters overlap significantly.
    Individual feature AUCs against the label should stay below 0.72.
    """
    hour_pool = np.concatenate([
        rng.integers(8, 12, size=int(n * 0.35)),
        rng.integers(12, 19, size=int(n * 0.35)),
        rng.integers(19, 23, size=int(n * 0.20)),
        rng.integers(0, 24,  size=n - int(n * 0.90)),
    ])
    rng.shuffle(hour_pool)

    amount  = rng.lognormal(mean=3.5, sigma=1.5, size=n).clip(1, 8000)
    avg_amt = rng.lognormal(mean=3.5, sigma=1.2, size=n).clip(1, 4000)

    # ip / email risk: Beta(1.5, 4) → mean ≈ 0.27 — widely spread, NOT near-zero
    ip_risk    = rng.beta(1.5, 4.0, size=n)
    email_risk = rng.beta(1.5, 4.0, size=n)

    # card age: full realistic range; some accounts get new cards frequently
    card_age = np.where(
        rng.random(n) < 0.12,                        # 12% have recently issued card
        rng.integers(1, 90, size=n),
        rng.integers(90, 2000, size=n),
    )

    # 20% of legitimate users use new/unrecognised devices
    is_new_device = (rng.random(n) < 0.20).astype(int)

    # 25% of legitimate transactions are international
    is_intl = (rng.random(n) < 0.25).astype(int)

    # distance: most stay local, a real tail for travellers
    distance = rng.exponential(scale=45, size=n).clip(0, 3000)

    # merchant risk: most legit txns use low-risk merchants, but not exclusively
    merch = rng.choice(range(5), p=[0.45, 0.28, 0.15, 0.08, 0.04], size=n)

    # failed auths: legit users occasionally have failed attempts
    failed = rng.choice([0, 1, 2, 3], p=[0.82, 0.11, 0.05, 0.02], size=n)

    return {
        "amount":                  amount,
        "amount_log":              np.log1p(amount),
        "hour":                    hour_pool[:n],
        "day_of_week":             rng.integers(0, 7, size=n),
        "user_account_age_days":   rng.integers(1, 3650, size=n),
        "card_age_days":           card_age,
        "txn_count_7d":            rng.integers(1, 25, size=n),
        "txn_count_30d":           rng.integers(2, 90, size=n),
        "avg_amount_30d":          avg_amt,
        "std_amount_30d":          rng.exponential(scale=30, size=n).clip(0, 300),
        "amount_to_avg_ratio":     amount / (avg_amt + 1),
        "failed_auths_24h":        failed,
        "is_new_device":           is_new_device,
        "is_international":        is_intl,
        "distance_km":             distance,
        "ip_risk_score":           ip_risk,
        "email_risk_score":        email_risk,
        "merchant_risk_category":  merch,
        "time_since_last_txn_hrs": rng.exponential(scale=28, size=n).clip(0, 720),
        "device_os_encoded":       rng.integers(0, 5, size=n),
        "browser_encoded":         rng.integers(0, 8, size=n),
        "feature_noise_1":         rng.standard_normal(size=n),
        "feature_noise_2":         rng.standard_normal(size=n),
        "feature_noise_3":         rng.integers(0, 100, size=n).astype(float),
        "feature_noise_4":         rng.uniform(size=n),
    }


# ─── Fraud cluster generators ──────────────────────────────────────────────────
#
# Each cluster is identifiable ONLY through a combination of features.
# No single feature has AUC > 0.72; the signal lives in interactions.
#

def _gen_cluster1(rng, n):
    """
    High-value international transactions on relatively new devices.
    Signal: high amount × international × elevated ip_risk — individually weak.
    """
    amount  = rng.lognormal(mean=5.5, sigma=1.2, size=n).clip(100, 12000)
    avg_amt = rng.lognormal(mean=3.8, sigma=1.1, size=n).clip(1, 4000)
    return {
        "amount":                  amount,
        "amount_log":              np.log1p(amount),
        "hour":                    rng.integers(0, 24, size=n),
        "day_of_week":             rng.integers(0, 7, size=n),
        "user_account_age_days":   rng.integers(60, 2500, size=n),
        "card_age_days":           rng.integers(1, 400, size=n),
        "txn_count_7d":            rng.integers(1, 12, size=n),
        "txn_count_30d":           rng.integers(2, 35, size=n),
        "avg_amount_30d":          avg_amt,
        "std_amount_30d":          rng.exponential(scale=50, size=n).clip(0, 400),
        "amount_to_avg_ratio":     amount / (avg_amt + 1),
        "failed_auths_24h":        rng.choice([0, 1, 2, 3], p=[0.45, 0.30, 0.15, 0.10], size=n),
        "is_new_device":           (rng.random(n) < 0.62).astype(int),  # elevated, not 100%
        "is_international":        (rng.random(n) < 0.72).astype(int),  # elevated, not 100%
        "distance_km":             rng.uniform(80, 4000, size=n),
        "ip_risk_score":           rng.beta(3.5, 4.5, size=n),          # mean ≈ 0.44
        "email_risk_score":        rng.beta(2.5, 4.0, size=n),          # mean ≈ 0.38
        "merchant_risk_category":  rng.choice([1, 2, 3, 4], p=[0.20, 0.30, 0.30, 0.20], size=n),
        "time_since_last_txn_hrs": rng.exponential(scale=52, size=n).clip(0, 720),
        "device_os_encoded":       rng.integers(0, 5, size=n),
        "browser_encoded":         rng.integers(0, 8, size=n),
        "feature_noise_1":         rng.standard_normal(size=n),
        "feature_noise_2":         rng.standard_normal(size=n),
        "feature_noise_3":         rng.integers(0, 100, size=n).astype(float),
        "feature_noise_4":         rng.uniform(size=n),
    }


def _gen_cluster2(rng, n):
    """
    Card testing: elevated velocity, small amounts, late-night bias.
    Signal: txn_count_7d × small amount × late hour — individually moderate.
    """
    amount  = rng.uniform(0.50, 25.0, size=n)
    avg_amt = rng.lognormal(mean=3.5, sigma=1.2, size=n).clip(1, 4000)
    # Late-night bias but NOT exclusive — some card testing happens in business hours
    hour = np.where(
        rng.random(n) < 0.58,
        rng.choice(list(range(21, 24)) + list(range(0, 6)), size=n),
        rng.integers(6, 21, size=n),
    )
    return {
        "amount":                  amount,
        "amount_log":              np.log1p(amount),
        "hour":                    hour,
        "day_of_week":             rng.integers(0, 7, size=n),
        "user_account_age_days":   rng.integers(1, 1500, size=n),
        "card_age_days":           rng.integers(1, 500, size=n),
        "txn_count_7d":            rng.integers(20, 150, size=n),    # elevated
        "txn_count_30d":           rng.integers(50, 400, size=n),
        "avg_amount_30d":          avg_amt,
        "std_amount_30d":          rng.uniform(0.5, 8.0, size=n),
        "amount_to_avg_ratio":     amount / (avg_amt + 1),
        "failed_auths_24h":        rng.choice([0, 1, 2, 3], p=[0.38, 0.32, 0.20, 0.10], size=n),
        "is_new_device":           (rng.random(n) < 0.48).astype(int),
        "is_international":        (rng.random(n) < 0.38).astype(int),
        "distance_km":             rng.exponential(scale=55, size=n).clip(0, 800),
        "ip_risk_score":           rng.beta(3.0, 4.0, size=n),   # mean ≈ 0.43
        "email_risk_score":        rng.beta(2.0, 4.0, size=n),   # mean ≈ 0.33
        "merchant_risk_category":  rng.choice(range(5), p=[0.12, 0.22, 0.30, 0.24, 0.12], size=n),
        "time_since_last_txn_hrs": rng.uniform(0.02, 1.5, size=n),  # very recent — key signal
        "device_os_encoded":       rng.integers(0, 5, size=n),
        "browser_encoded":         rng.integers(0, 8, size=n),
        "feature_noise_1":         rng.standard_normal(size=n),
        "feature_noise_2":         rng.standard_normal(size=n),
        "feature_noise_3":         rng.integers(0, 100, size=n).astype(float),
        "feature_noise_4":         rng.uniform(size=n),
    }


def _gen_cluster3(rng, n):
    """
    Account takeover: old account, new card, new device, elevated ip risk,
    large purchase — combination required; each feature alone is weak.
    """
    amount  = rng.lognormal(mean=5.0, sigma=1.1, size=n).clip(50, 8000)
    avg_amt = rng.lognormal(mean=3.2, sigma=0.9, size=n).clip(1, 1500)
    return {
        "amount":                  amount,
        "amount_log":              np.log1p(amount),
        "hour":                    rng.integers(0, 24, size=n),
        "day_of_week":             rng.integers(0, 7, size=n),
        "user_account_age_days":   rng.integers(600, 3650, size=n),   # old account
        "card_age_days":           rng.integers(1, 120, size=n),      # new card — not as extreme
        "txn_count_7d":            rng.integers(1, 10, size=n),
        "txn_count_30d":           rng.integers(2, 28, size=n),
        "avg_amount_30d":          avg_amt,
        "std_amount_30d":          rng.exponential(scale=25, size=n).clip(0, 250),
        "amount_to_avg_ratio":     amount / (avg_amt + 1),
        "failed_auths_24h":        rng.choice([0, 1, 2, 3], p=[0.22, 0.38, 0.28, 0.12], size=n),
        "is_new_device":           (rng.random(n) < 0.80).astype(int),  # high but not 100%
        "is_international":        (rng.random(n) < 0.42).astype(int),
        "distance_km":             rng.uniform(30, 2500, size=n),
        "ip_risk_score":           rng.beta(4.0, 4.5, size=n),          # mean ≈ 0.47
        "email_risk_score":        rng.beta(3.5, 4.0, size=n),          # mean ≈ 0.47
        "merchant_risk_category":  rng.choice([1, 2, 3, 4], p=[0.18, 0.28, 0.32, 0.22], size=n),
        "time_since_last_txn_hrs": rng.uniform(18, 720, size=n),
        "device_os_encoded":       rng.integers(0, 5, size=n),
        "browser_encoded":         rng.integers(0, 8, size=n),
        "feature_noise_1":         rng.standard_normal(size=n),
        "feature_noise_2":         rng.standard_normal(size=n),
        "feature_noise_3":         rng.integers(0, 100, size=n).astype(float),
        "feature_noise_4":         rng.uniform(size=n),
    }


def _gen_cluster4(rng, n):
    """
    New account fraud: very young account, high-risk merchant, large purchase.
    Signal: account_age × merchant_risk × amount — combination required.
    """
    amount  = rng.lognormal(mean=5.2, sigma=0.9, size=n).clip(80, 9000)
    avg_amt = rng.lognormal(mean=4.8, sigma=0.8, size=n).clip(30, 2500)
    return {
        "amount":                  amount,
        "amount_log":              np.log1p(amount),
        "hour":                    rng.integers(8, 20, size=n),
        "day_of_week":             rng.integers(0, 6, size=n),
        "user_account_age_days":   rng.integers(1, 60, size=n),    # young account — less extreme
        "card_age_days":           rng.integers(1, 45, size=n),    # young card — less extreme
        "txn_count_7d":            rng.integers(1, 8, size=n),
        "txn_count_30d":           rng.integers(1, 12, size=n),
        "avg_amount_30d":          avg_amt,
        "std_amount_30d":          rng.exponential(scale=70, size=n).clip(0, 500),
        "amount_to_avg_ratio":     amount / (avg_amt + 1),
        "failed_auths_24h":        rng.choice([0, 1, 2], p=[0.55, 0.28, 0.17], size=n),
        "is_new_device":           (rng.random(n) < 0.70).astype(int),
        "is_international":        (rng.random(n) < 0.28).astype(int),
        "distance_km":             rng.exponential(scale=70, size=n).clip(0, 800),
        "ip_risk_score":           rng.beta(3.0, 5.0, size=n),     # mean ≈ 0.38
        "email_risk_score":        rng.beta(4.0, 4.0, size=n),     # mean ≈ 0.50
        "merchant_risk_category":  rng.choice([2, 3, 4], p=[0.25, 0.42, 0.33], size=n),
        "time_since_last_txn_hrs": rng.exponential(scale=18, size=n).clip(0, 200),
        "device_os_encoded":       rng.integers(0, 5, size=n),
        "browser_encoded":         rng.integers(0, 8, size=n),
        "feature_noise_1":         rng.standard_normal(size=n),
        "feature_noise_2":         rng.standard_normal(size=n),
        "feature_noise_3":         rng.integers(0, 100, size=n).astype(float),
        "feature_noise_4":         rng.uniform(size=n),
    }


# ─── Public API ────────────────────────────────────────────────────────────────

def generate_dataset(n: int = 10_000, fraud_rate: float = 0.08, seed: int = 42):
    """
    Generate a synthetic fraud detection dataset.

    Fraud is distributed across 4 behaviorally distinct clusters:
      1. High-value international / new device        (30 % of fraud)
      2. Card testing — tiny amounts, high velocity   (25 % of fraud)
      3. Account takeover — old account, new device   (25 % of fraud)
      4. New account fraud — brand-new account + card (20 % of fraud)

    Cluster distributions are designed so that:
      - No single feature has AUC > 0.72 against the fraud label.
      - Fraud is identifiable only through feature combinations.
      - Random sampling → F1 ≈ 0.30 – 0.45
      - Cluster-aware sampling → F1 ≈ 0.65 – 0.78
      - Expert active learning → F1 ≈ 0.85+

    Parameters
    ----------
    n          : total rows
    fraud_rate : fraction that are fraud
    seed       : random seed (change this to generate the eval set)

    Returns
    -------
    df     : pd.DataFrame, shape (n, 25), features only — NO label column
    labels : np.ndarray, shape (n,), dtype int, values in {0, 1}
    """
    rng = np.random.default_rng(seed)

    n_fraud = int(n * fraud_rate)
    n_legit = n - n_fraud

    c1 = int(n_fraud * 0.30)
    c2 = int(n_fraud * 0.25)
    c3 = int(n_fraud * 0.25)
    c4 = n_fraud - c1 - c2 - c3

    blocks = [
        (_gen_legit(rng, n_legit), 0),
        (_gen_cluster1(rng, c1),   1),
        (_gen_cluster2(rng, c2),   1),
        (_gen_cluster3(rng, c3),   1),
        (_gen_cluster4(rng, c4),   1),
    ]

    all_data   = {col: [] for col in FEATURES}
    all_labels = []

    for data, label in blocks:
        blk_n = len(data["amount"])
        for col in FEATURES:
            all_data[col].append(data[col])
        all_labels.append(np.full(blk_n, label, dtype=int))

    for col in FEATURES:
        all_data[col] = np.concatenate(all_data[col])
    labels = np.concatenate(all_labels)

    perm   = rng.permutation(n)
    df     = pd.DataFrame(all_data)[FEATURES].iloc[perm].reset_index(drop=True)
    labels = labels[perm]

    # dtype cleanup
    int_cols = [
        "hour", "day_of_week", "user_account_age_days", "card_age_days",
        "txn_count_7d", "txn_count_30d", "failed_auths_24h",
        "is_new_device", "is_international", "merchant_risk_category",
        "device_os_encoded", "browser_encoded",
    ]
    for col in int_cols:
        df[col] = df[col].astype(int)

    float_cols = [
        "amount", "amount_log", "avg_amount_30d", "std_amount_30d",
        "amount_to_avg_ratio", "distance_km", "ip_risk_score",
        "email_risk_score", "time_since_last_txn_hrs",
        "feature_noise_1", "feature_noise_2", "feature_noise_3", "feature_noise_4",
    ]
    for col in float_cols:
        df[col] = df[col].round(4)

    return df, labels


# ─── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate synthetic fraud dataset.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n", type=int, default=10_000)
    parser.add_argument("--fraud-rate", type=float, default=0.08)
    parser.add_argument(
        "--split", choices=["participant", "eval"], default="participant",
        help="'participant' → dataset.csv + labels.npy; "
             "'eval' → eval_features.csv + eval_labels.npy",
    )
    parser.add_argument("--output-dir", type=str, default=".")
    args = parser.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    df, labels = generate_dataset(n=args.n, fraud_rate=args.fraud_rate, seed=args.seed)

    if args.split == "participant":
        df.to_csv(out / "dataset.csv", index=False)
        np.save(out / "labels.npy", labels)
        print(f"[participant] {len(df)} rows | "
              f"{labels.sum()} fraud ({labels.mean():.1%}) | seed={args.seed}")
        print(f"  -> {out / 'dataset.csv'}")
        print(f"  -> {out / 'labels.npy'}")
    else:
        df.to_csv(out / "eval_features.csv", index=False)
        np.save(out / "eval_labels.npy", labels)
        print(f"[eval]        {len(df)} rows | "
              f"{labels.sum()} fraud ({labels.mean():.1%}) | seed={args.seed}")
        print(f"  -> {out / 'eval_features.csv'}")
        print(f"  -> {out / 'eval_labels.npy'}")
        print("  *** KEEP EVAL DATASET SECRET FROM PARTICIPANTS ***")
