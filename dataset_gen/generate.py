"""
Synthetic talent-fraud detection dataset generator.
Sponsored problem by Eightfold AI.

Usage:
    python generate.py --split participant --seed 42 --output-dir ../participant_kit
    python generate.py --split eval      --seed 99 --output-dir ../eval_dataset

Domain
------
Eightfold AI's talent platform processes millions of candidate profiles and
job applications. A small fraction of these are fraudulent — fake credentials,
inflated experience, bot-generated spam applications, and hijacked accounts.

Design rationale
----------------
Each fraud cluster shifts ONLY 3-4 features from the legitimate baseline.
Non-signal features remain at the SAME distribution as legitimate profiles,
so no single feature is a strong global discriminator.

Within each cluster, the signal features ARE strongly shifted (within-cluster
AUC 0.85-0.97 vs legit). This means:

  • ~2 fraud labels per cluster  (random sampling)  → model can't learn ANY cluster → F1 ≈ 0.20-0.35
  • ~10 fraud labels per cluster (smart sampling)   → model detects ALL clusters  → F1 ≈ 0.65-0.78
  • 20+ fraud labels per cluster (expert sampling)  → near-optimal detection      → F1 ≈ 0.85+

Global (per-feature) AUC stays low (≤ 0.72) because each feature is elevated
for at most 1 cluster (25-30% of all fraud); the remaining fraud look like
legitimate profiles in that feature.

CRITICAL: Each cluster's signal features are COMPLETELY NON-OVERLAPPING.
This is what creates the sharp gradient: random sampling yields ~2 fraud per
cluster, which is not enough to learn any cluster reliably. Smart active learning
that finds 10+ per cluster unlocks all four detection patterns simultaneously.

Fraud clusters
--------------
  C1 — Credential Fraud        : fake degrees / ghost companies
  C2 — Application Bombing     : bot-driven mass-apply spam
  C3 — Account Takeover        : hijacked legitimate profile
  C4 — Ghost Profile / Synthetic Identity : freshly fabricated identity
"""

import numpy as np
import pandas as pd
from pathlib import Path

FEATURES = [
    "profile_age_days", "applications_7d", "applications_30d",
    "avg_applications_30d", "app_to_avg_ratio",
    "skills_count", "endorsements_count",
    "experience_years", "skills_to_exp_ratio",
    "institution_risk_score", "company_risk_score",
    "gpa_anomaly_score", "tenure_gap_months", "avg_tenure_months",
    "time_since_last_app_hrs",
    "is_new_device", "ip_risk_score", "email_risk_score",
    "login_velocity_24h", "failed_logins_24h",
    "copy_paste_ratio",
    "feature_noise_1", "feature_noise_2", "feature_noise_3", "feature_noise_4",
]


# ─── Shared baseline (legit distribution) ────────────────────────────────────

def _base(rng, n):
    """
    Legitimate-candidate feature distributions. Both legit rows AND the
    non-signal features of fraud rows are drawn from this.
    """
    avg_apps = rng.lognormal(mean=1.2, sigma=0.8, size=n).clip(0.1, 30)
    apps_7d  = rng.integers(0, 8, size=n)
    apps_30d = apps_7d + rng.integers(0, 20, size=n)

    exp_yrs  = rng.lognormal(mean=1.8, sigma=0.9, size=n).clip(0, 40)
    skills   = (exp_yrs * rng.uniform(1.5, 3.0, size=n) + rng.integers(0, 10, size=n)).clip(1, 80)

    return {
        "profile_age_days":       rng.integers(30, 3650, size=n),
        "applications_7d":        apps_7d,
        "applications_30d":       apps_30d,
        "avg_applications_30d":   avg_apps,
        "app_to_avg_ratio":       apps_30d / (avg_apps * 30 + 1),
        "skills_count":           skills.astype(int),
        "endorsements_count":     rng.integers(0, 60, size=n),
        "experience_years":       exp_yrs.round(1),
        "skills_to_exp_ratio":    (skills / (exp_yrs + 1)).round(3),
        "institution_risk_score": rng.beta(1.5, 5.0, size=n),        # mostly low
        "company_risk_score":     rng.beta(1.5, 5.0, size=n),
        "gpa_anomaly_score":      rng.beta(1.5, 5.0, size=n),        # 0=plausible, 1=very suspicious
        "tenure_gap_months":      rng.exponential(scale=3, size=n).clip(0, 60).round(1),
        "avg_tenure_months":      rng.lognormal(mean=3.0, sigma=0.7, size=n).clip(1, 120).round(1),
        "time_since_last_app_hrs":rng.lognormal(mean=3.5, sigma=1.5, size=n).clip(0.1, 720).round(2),
        "is_new_device":          (rng.random(n) < 0.15).astype(int),
        "ip_risk_score":          rng.beta(1.5, 5.0, size=n),
        "email_risk_score":       rng.beta(1.5, 5.0, size=n),
        "login_velocity_24h":     rng.poisson(lam=1.5, size=n).clip(0, 20),
        "failed_logins_24h":      rng.poisson(lam=0.3, size=n).clip(0, 5),
        "copy_paste_ratio":       rng.beta(2.0, 6.0, size=n),        # mostly low
        "feature_noise_1":        rng.standard_normal(size=n),
        "feature_noise_2":        rng.standard_normal(size=n),
        "feature_noise_3":        rng.integers(0, 100, size=n).astype(float),
        "feature_noise_4":        rng.uniform(size=n),
    }


# ─── Fraud cluster generators ─────────────────────────────────────────────────

def _gen_legit(rng, n):
    return _base(rng, n)


def _gen_cluster1(rng, n):
    """
    Credential Fraud: fake degrees, unverifiable institutions, ghost companies.
    EXCLUSIVE signals: institution_risk_score ↑↑, gpa_anomaly_score ↑↑,
                       company_risk_score ↑↑, tenure_gap_months ↑↑.
    PARTIAL PREVALENCE: 60% of C1 fraud rows carry these signals.

    Fraudsters claim prestigious institutions/companies that don't check out.
    Their GPAs are suspiciously perfect, and employment history has large gaps.

    Global AUC ≈ 0.57; NO overlap with C2/C3/C4 signals.
    """
    d   = _base(rng, n)
    sig = rng.random(n) < 0.60

    d["institution_risk_score"] = np.where(sig,
                                           rng.beta(7.0, 2.5, size=n),   # mean≈0.74
                                           d["institution_risk_score"])
    d["gpa_anomaly_score"]      = np.where(sig,
                                           rng.beta(6.5, 2.5, size=n),   # mean≈0.72
                                           d["gpa_anomaly_score"])
    d["company_risk_score"]     = np.where(sig,
                                           rng.beta(6.0, 3.0, size=n),   # mean≈0.67
                                           d["company_risk_score"])
    d["tenure_gap_months"]      = np.where(sig,
                                           rng.exponential(scale=18, size=n).clip(6, 60),
                                           d["tenure_gap_months"])
    return d


def _gen_cluster2(rng, n):
    """
    Application Bombing: bots or click-farms mass-applying to every open role.
    EXCLUSIVE signals: applications_7d ↑↑, applications_30d ↑↑,
                       app_to_avg_ratio ↑↑, time_since_last_app_hrs ↓↓.
    PARTIAL PREVALENCE: 60% of C2 fraud rows carry these signals.

    Bot accounts send dozens of applications per day with near-zero idle time.

    Global AUC ≈ 0.57; NO overlap with C1/C3/C4 signals.
    """
    d   = _base(rng, n)
    sig = rng.random(n) < 0.60

    apps_7d  = np.where(sig, rng.integers(40, 150, size=n), d["applications_7d"])
    apps_30d = np.where(sig, rng.integers(120, 450, size=n), d["applications_30d"])
    d["applications_7d"]          = apps_7d
    d["applications_30d"]         = apps_30d
    d["app_to_avg_ratio"]         = np.where(sig,
                                             apps_30d / (d["avg_applications_30d"] * 30 + 1),
                                             d["app_to_avg_ratio"])
    d["time_since_last_app_hrs"]  = np.where(sig,
                                             rng.uniform(0.005, 0.5, size=n),
                                             d["time_since_last_app_hrs"])
    return d


def _gen_cluster3(rng, n):
    """
    Account Takeover: attacker hijacks a legitimate candidate's profile to
    apply for roles (or extract recruiter contacts / salary data).
    EXCLUSIVE signals: is_new_device ↑↑, failed_logins_24h ↑↑,
                       login_velocity_24h ↑↑, email_risk_score ↑↑.
    PARTIAL PREVALENCE: 60% of C3 fraud rows carry these signals.

    Attackers log in from unfamiliar devices, fail auth several times,
    and then interact at high velocity once in.

    Global AUC ≈ 0.56; NO overlap with C1/C2/C4 signals.
    """
    d   = _base(rng, n)
    sig = rng.random(n) < 0.60

    d["is_new_device"]      = np.where(sig, (rng.random(n) < 0.95).astype(int), d["is_new_device"])
    d["failed_logins_24h"]  = np.where(sig,
                                       rng.poisson(lam=3.5, size=n).clip(0, 5),
                                       d["failed_logins_24h"])
    d["login_velocity_24h"] = np.where(sig,
                                       rng.integers(10, 40, size=n),
                                       d["login_velocity_24h"])
    d["email_risk_score"]   = np.where(sig,
                                       rng.beta(7.0, 2.5, size=n),    # mean≈0.74
                                       d["email_risk_score"])
    return d


def _gen_cluster4(rng, n):
    """
    Ghost Profile / Synthetic Identity: a freshly-created fake account with
    an implausibly polished profile (copy-pasted text, inflated skills list).
    EXCLUSIVE signals: profile_age_days ↓↓, copy_paste_ratio ↑↑,
                       skills_to_exp_ratio ↑↑ (too many skills for claimed exp).
    PARTIAL PREVALENCE: 60% of C4 fraud rows carry these signals.

    New fake profiles appear highly complete but the ratio of claimed skills
    to experience is statistically abnormal, and profile text is recycled.

    Global AUC ≈ 0.55; NO overlap with C1/C2/C3 signals.
    """
    d   = _base(rng, n)
    sig = rng.random(n) < 0.60

    # Very fresh account
    new_profile = rng.random(n) < 0.95
    young_days  = np.where(new_profile, rng.integers(1, 14, size=n), rng.integers(14, 90, size=n))
    d["profile_age_days"]     = np.where(sig, young_days, d["profile_age_days"])

    # Suspiciously high copy-paste ratio
    d["copy_paste_ratio"]     = np.where(sig,
                                         rng.beta(7.0, 2.5, size=n),   # mean≈0.74
                                         d["copy_paste_ratio"])

    # Many skills relative to thin experience
    d["skills_to_exp_ratio"]  = np.where(sig,
                                         rng.lognormal(mean=3.5, sigma=0.6, size=n).clip(15, 80),
                                         d["skills_to_exp_ratio"])
    return d


# ─── Public API ───────────────────────────────────────────────────────────────

def generate_dataset(n: int = 10_000, fraud_rate: float = 0.08, seed: int = 42):
    """
    Returns
    -------
    df     : pd.DataFrame (n, 25) — features only, NO label column
    labels : np.ndarray  (n,)    — 0 = legitimate candidate, 1 = fraudulent
    """
    rng     = np.random.default_rng(seed)
    n_fraud = int(n * fraud_rate)
    n_legit = n - n_fraud

    c1 = int(n_fraud * 0.30)   # credential fraud
    c2 = int(n_fraud * 0.25)   # application bombing
    c3 = int(n_fraud * 0.25)   # account takeover
    c4 = n_fraud - c1 - c2 - c3  # ghost profile

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
        labels.append(np.full(len(data["profile_age_days"]), lbl, dtype=int))

    for f in FEATURES:
        cols[f] = np.concatenate(cols[f])
    labels = np.concatenate(labels)

    perm   = rng.permutation(n)
    df     = pd.DataFrame(cols)[FEATURES].iloc[perm].reset_index(drop=True)
    labels = labels[perm]

    int_cols = [
        "profile_age_days", "applications_7d", "applications_30d",
        "skills_count", "endorsements_count",
        "login_velocity_24h", "failed_logins_24h",
        "is_new_device",
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
