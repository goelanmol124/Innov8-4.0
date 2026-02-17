"""
Synthetic fraud detection dataset generator.

Usage:
    python generate.py --split participant --seed 42 --output-dir ../participant_kit
    python generate.py --split eval      --seed 99 --output-dir ../eval_dataset

Design rationale
----------------
Each fraud cluster shifts ONLY 3-4 features from the legitimate baseline.
Non-signal features remain at the SAME distribution as legitimate transactions,
so no single feature is a strong global discriminator.

Within each cluster, the signal features ARE strongly shifted (within-cluster
AUC 0.85-0.97 vs legit). This means:

  • ~2 fraud labels per cluster  (random sampling)  → model can't learn ANY cluster → F1 ≈ 0.20-0.35
  • ~10 fraud labels per cluster (smart sampling)   → model detects ALL clusters  → F1 ≈ 0.65-0.78
  • 20+ fraud labels per cluster (expert sampling)  → near-optimal detection      → F1 ≈ 0.85+

Global (per-feature) AUC stays low (≤ 0.72) because each feature is elevated
for at most 1 cluster (25-30% of all fraud); the remaining fraud look like
legitimate in that feature.

CRITICAL: Each cluster's signal features are COMPLETELY NON-OVERLAPPING.
This is what creates the sharp gradient: random sampling yields ~2 fraud per
cluster, which is not enough to learn any cluster reliably. Smart active learning
that finds 10+ per cluster unlocks all four detection patterns simultaneously.
"""

import numpy as np
import pandas as pd
from pathlib import Path

FEATURES = [
    "amount", "amount_log", "hour", "day_of_week",
    "user_account_age_days", "card_age_days",
    "txn_count_7d", "txn_count_30d",
    "avg_amount_30d", "std_amount_30d", "amount_to_avg_ratio",
    "failed_auths_24h", "is_new_device", "is_international",
    "distance_km", "ip_risk_score", "email_risk_score",
    "merchant_risk_category", "time_since_last_txn_hrs",
    "device_os_encoded", "browser_encoded",
    "feature_noise_1", "feature_noise_2", "feature_noise_3", "feature_noise_4",
]


# ─── Shared baseline (legit distribution) ────────────────────────────────────

def _base(rng, n):
    """
    Legitimate-user feature distributions. Both legit rows AND the
    non-signal features of fraud rows are drawn from this.
    """
    hour_pool = np.concatenate([
        rng.integers(8, 12,  size=int(n * 0.35)),
        rng.integers(12, 19, size=int(n * 0.35)),
        rng.integers(19, 23, size=int(n * 0.20)),
        rng.integers(0, 24,  size=n - int(n * 0.90)),
    ])
    rng.shuffle(hour_pool)

    # Card age: 30% recently issued, 70% established
    fresh    = rng.random(n) < 0.30
    card_age = np.where(fresh, rng.integers(1, 300, n), rng.integers(300, 1800, n))

    amt     = rng.lognormal(mean=3.5, sigma=1.8, size=n).clip(0.5, 10000)
    avg_amt = rng.lognormal(mean=3.5, sigma=1.3, size=n).clip(0.5, 5000)

    return {
        "amount":                  amt,
        "amount_log":              np.log1p(amt),
        "hour":                    hour_pool[:n],
        "day_of_week":             rng.integers(0, 7,    size=n),
        "user_account_age_days":   rng.integers(1, 3650, size=n),
        "card_age_days":           card_age,
        "txn_count_7d":            rng.integers(1, 25,   size=n),
        "txn_count_30d":           rng.integers(3, 80,   size=n),
        "avg_amount_30d":          avg_amt,
        "std_amount_30d":          rng.exponential(scale=30, size=n).clip(0, 350),
        "amount_to_avg_ratio":     amt / (avg_amt + 1),
        # Beta(1.5,5) → mean≈0.23, spreads [0,1] but mostly low
        "ip_risk_score":           rng.beta(1.5, 5.0, size=n),
        "email_risk_score":        rng.beta(1.5, 5.0, size=n),
        # Poisson(0.3) → ~74% zero, ~22% one, ~4% two+
        "failed_auths_24h":        rng.poisson(lam=0.3, size=n).clip(0, 4),
        "is_new_device":           (rng.random(n) < 0.20).astype(int),
        "is_international":        (rng.random(n) < 0.25).astype(int),
        "distance_km":             rng.exponential(scale=50, size=n).clip(0, 4000),
        "merchant_risk_category":  rng.choice(5, p=[0.35, 0.30, 0.20, 0.10, 0.05], size=n),
        "time_since_last_txn_hrs": rng.lognormal(mean=2.0, sigma=1.5, size=n).clip(0.01, 720),
        "device_os_encoded":       rng.integers(0, 5, size=n),
        "browser_encoded":         rng.integers(0, 8, size=n),
        "feature_noise_1":         rng.standard_normal(size=n),
        "feature_noise_2":         rng.standard_normal(size=n),
        "feature_noise_3":         rng.integers(0, 100, size=n).astype(float),
        "feature_noise_4":         rng.uniform(size=n),
    }


# ─── Fraud cluster generators ─────────────────────────────────────────────────

def _gen_legit(rng, n):
    return _base(rng, n)


def _gen_cluster1(rng, n):
    """
    High-value international fraud.
    EXCLUSIVE signals: amount ↑↑, is_international ↑↑, distance_km ↑↑, ip_risk_score ↑↑.
    PARTIAL PREVALENCE: only 60% of C1 fraud rows have these signals elevated.
    The remaining 40% draw from the base (legit) distribution, making them
    indistinguishable from legitimate transactions on individual features.

    This requires ~10+ labeled C1 samples to reliably detect the cluster;
    with only 2 samples, expected 1.2 have the signal — not enough to generalise.

    Global AUC (30% cluster weight, 60% prevalence):
        each feature ≈ 0.30×0.60×0.90 + 0.70×0.50 + 0.30×0.40×0.50 ≈ 0.57
    NO overlap with C2/C3/C4 signals.
    """
    d   = _base(rng, n)
    sig = rng.random(n) < 0.60  # only 60% have the cluster signal

    amt                        = np.where(sig,
                                          rng.lognormal(mean=6.0, sigma=1.0, size=n).clip(200, 30000),
                                          d["amount"])
    d["amount"]                = amt
    d["amount_log"]            = np.log1p(amt)
    d["amount_to_avg_ratio"]   = amt / (d["avg_amount_30d"] + 1)
    d["is_international"]      = np.where(sig,
                                          (rng.random(n) < 0.92).astype(int),
                                          d["is_international"])
    d["distance_km"]           = np.where(sig,
                                          rng.exponential(scale=600, size=n).clip(0, 4000),
                                          d["distance_km"])
    d["ip_risk_score"]         = np.where(sig,
                                          rng.beta(6.0, 3.0, size=n),   # mean≈0.67
                                          d["ip_risk_score"])
    return d


def _gen_cluster2(rng, n):
    """
    Card testing: micro-amounts, extreme velocity, back-to-back transactions.
    EXCLUSIVE signals: amount ↓↓, txn_count_7d ↑↑, txn_count_30d ↑↑, time_since ↓↓.
    PARTIAL PREVALENCE: 60% of C2 fraud rows have these signals.

    Global AUC (25% cluster weight, 60% prevalence):
        each feature ≈ 0.25×0.60×0.95 + 0.75×0.50 + 0.25×0.40×0.50 ≈ 0.57
    NO overlap with C1/C3/C4 signals.
    """
    d   = _base(rng, n)
    sig = rng.random(n) < 0.60

    amt                          = np.where(sig, rng.uniform(0.5, 12, size=n), d["amount"])
    d["amount"]                  = amt
    d["amount_log"]              = np.log1p(amt)
    d["amount_to_avg_ratio"]     = amt / (d["avg_amount_30d"] + 1)
    d["txn_count_7d"]            = np.where(sig, rng.integers(40, 150, size=n), d["txn_count_7d"])
    d["txn_count_30d"]           = np.where(sig, rng.integers(120, 400, size=n), d["txn_count_30d"])
    d["time_since_last_txn_hrs"] = np.where(sig,
                                            rng.uniform(0.005, 0.25, size=n),
                                            d["time_since_last_txn_hrs"])
    return d


def _gen_cluster3(rng, n):
    """
    Account takeover: new device, high email risk, repeated failed auths.
    EXCLUSIVE signals: is_new_device ↑↑, email_risk_score ↑↑, failed_auths_24h ↑↑.
    PARTIAL PREVALENCE: 60% of C3 fraud rows have these signals.

    Global AUC (25% cluster weight, 60% prevalence):
        each feature ≈ 0.25×0.60×0.92 + 0.75×0.50 + 0.25×0.40×0.50 ≈ 0.56
    NO overlap with C1/C2/C4 signals (ip_risk stays at base).
    """
    d   = _base(rng, n)
    sig = rng.random(n) < 0.60

    d["is_new_device"]    = np.where(sig, (rng.random(n) < 0.95).astype(int), d["is_new_device"])
    d["email_risk_score"] = np.where(sig, rng.beta(7.0, 2.5, size=n), d["email_risk_score"])
    d["failed_auths_24h"] = np.where(sig,
                                     rng.poisson(lam=3.0, size=n).clip(0, 4),
                                     d["failed_auths_24h"])
    return d


def _gen_cluster4(rng, n):
    """
    New-account fraud: fresh identity, immediate high-risk purchase.
    EXCLUSIVE signals: merchant_risk_category ↑↑, user_account_age_days ↓↓, card_age_days ↓↓.
    PARTIAL PREVALENCE: 60% of C4 fraud rows have these signals.

    Global AUC (20% cluster weight, 60% prevalence):
        each feature ≈ 0.20×0.60×0.93 + 0.80×0.50 + 0.20×0.40×0.50 ≈ 0.55
    NO overlap with C1/C2/C3 signals (is_new_device, email_risk stay at base).
    """
    d   = _base(rng, n)
    sig = rng.random(n) < 0.60

    d["merchant_risk_category"] = np.where(sig,
                                            rng.choice([3, 4], p=[0.45, 0.55], size=n),
                                            d["merchant_risk_category"])

    new_acct = rng.random(n) < 0.95
    young_acct = np.where(new_acct, rng.integers(1, 14, size=n), rng.integers(14, 365, size=n))
    d["user_account_age_days"]  = np.where(sig, young_acct, d["user_account_age_days"])

    new_card = rng.random(n) < 0.95
    young_card = np.where(new_card, rng.integers(1, 21, size=n), rng.integers(21, 365, size=n))
    d["card_age_days"]          = np.where(sig, young_card, d["card_age_days"])

    return d


# ─── Public API ───────────────────────────────────────────────────────────────

def generate_dataset(n: int = 10_000, fraud_rate: float = 0.08, seed: int = 42):
    """
    Returns
    -------
    df     : pd.DataFrame (n, 25) — features only, NO label column
    labels : np.ndarray  (n,)    — 0 = legitimate, 1 = fraud
    """
    rng     = np.random.default_rng(seed)
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

    cols, labels = {f: [] for f in FEATURES}, []
    for data, lbl in blocks:
        for f in FEATURES:
            cols[f].append(data[f])
        labels.append(np.full(len(data["amount"]), lbl, dtype=int))

    for f in FEATURES:
        cols[f] = np.concatenate(cols[f])
    labels = np.concatenate(labels)

    perm   = rng.permutation(n)
    df     = pd.DataFrame(cols)[FEATURES].iloc[perm].reset_index(drop=True)
    labels = labels[perm]

    int_cols = [
        "hour", "day_of_week", "user_account_age_days", "card_age_days",
        "txn_count_7d", "txn_count_30d", "failed_auths_24h",
        "is_new_device", "is_international", "merchant_risk_category",
        "device_os_encoded", "browser_encoded",
    ]
    for c in int_cols:
        df[c] = df[c].astype(int)
    for c in set(FEATURES) - set(int_cols):
        df[c] = df[c].round(4)

    return df, labels


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--seed",       type=int,   default=42)
    p.add_argument("--n",          type=int,   default=10_000)
    p.add_argument("--fraud-rate", type=float, default=0.08)
    p.add_argument("--split",      choices=["participant", "eval"], default="participant")
    p.add_argument("--output-dir", type=str,   default=".")
    args = p.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    df, labels = generate_dataset(n=args.n, fraud_rate=args.fraud_rate, seed=args.seed)

    if args.split == "participant":
        df.to_csv(out / "dataset.csv", index=False)
        np.save(out / "labels.npy", labels)
        print(f"[participant] {len(df)} rows | {labels.sum()} fraud ({labels.mean():.1%}) | seed={args.seed}")
    else:
        df.to_csv(out / "eval_features.csv", index=False)
        np.save(out / "eval_labels.npy", labels)
        print(f"[eval]  {len(df)} rows | {labels.sum()} fraud ({labels.mean():.1%}) | seed={args.seed}")
        print("  *** KEEP EVAL DATASET SECRET ***")
