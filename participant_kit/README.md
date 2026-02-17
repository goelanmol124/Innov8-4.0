# Innov8 4.0 — Problem Statement

## The Challenge: Fraud Detection with a Limited Oracle

You are given a dataset of **10,000 financial transactions** (features only, no labels).
Your task is to build a classifier that predicts whether each transaction is fraudulent.

The catch: you can only **query the ground-truth label for up to 100 rows**.

Every query you make costs budget. How you spend that budget determines how well your model learns.

---

## Dataset

`dataset.csv` — 10,000 rows, 25 features, no label column.

| Feature | Description |
|---|---|
| `amount` | Transaction amount (USD) |
| `amount_log` | log(1 + amount) |
| `hour` | Hour of day (0–23) |
| `day_of_week` | 0=Monday … 6=Sunday |
| `user_account_age_days` | Age of user account in days |
| `card_age_days` | Age of card used in days |
| `txn_count_7d` | Transactions in last 7 days |
| `txn_count_30d` | Transactions in last 30 days |
| `avg_amount_30d` | Average transaction amount over last 30 days |
| `std_amount_30d` | Std deviation of transaction amount over last 30 days |
| `amount_to_avg_ratio` | amount / avg_amount_30d |
| `failed_auths_24h` | Failed authentication attempts in last 24 hours |
| `is_new_device` | 1 if device not previously seen for this account |
| `is_international` | 1 if transaction is cross-border |
| `distance_km` | Distance from account's home location (km) |
| `ip_risk_score` | IP address risk score (0–1) |
| `email_risk_score` | Email address risk score (0–1) |
| `merchant_risk_category` | Merchant risk tier (0=lowest … 4=highest) |
| `time_since_last_txn_hrs` | Hours since previous transaction |
| `device_os_encoded` | Device OS (encoded 0–4) |
| `browser_encoded` | Browser (encoded 0–7) |
| `feature_noise_1` | — |
| `feature_noise_2` | — |
| `feature_noise_3` | — |
| `feature_noise_4` | — |

Approximately **8% of transactions are fraudulent**.

---

## Your Task

Implement the `run_agent` function in `agent.py`:

```python
def run_agent(df: pd.DataFrame, oracle_fn, budget: int) -> np.ndarray:
    """
    df        : 10,000-row DataFrame of features (no labels).
    oracle_fn : oracle_fn([i1, i2, ...]) → [label1, label2, ...]
                Total indices across ALL calls must not exceed budget.
    budget    : 100

    Returns   : array of shape (10000,) with predictions in {0, 1}.
                Partial predictions (shorter arrays) are padded with 0.
    """
```

**You may call `oracle_fn` however you like** — one large call, many small calls, iteratively — as long as the total number of requested indices does not exceed 100.

---

## Scoring

Your submission is evaluated on **F1 score on the positive class (fraud)** on a held-out dataset with the same structure but different data.

**Tie-breaking** (applied in this order when F1 scores are equal):
1. **Fewer oracle queries used** (more efficient sampling → better rank)
2. **Faster runtime** (lower wall-clock time → better rank)

---

## Allowed Libraries

Your code may only import from the following:

```
numpy, pandas, sklearn (scikit-learn), scipy,
math, random, statistics, collections, itertools,
functools, typing, warnings, copy, time, json, re
```

**Any other import will result in disqualification.**
In particular, these are banned: `os`, `sys`, `subprocess`, `socket`, `requests`, `urllib`, `pickle`, `gc`, `inspect`, `ctypes`, `importlib`, `threading`, `multiprocessing`.

---

## Testing Locally

```bash
# Install dependencies
pip install -r requirements.txt

# Test the provided baseline agent
python framework.py --agent example_agent.py

# Test your own agent
python framework.py --agent agent.py
```

The local score is computed against `dataset.csv`. The **final evaluation uses a different dataset** with the same 25 features and similar statistics — so local and final scores will differ slightly.

---

## Submission Format

Submit a **ZIP file** containing at minimum:

```
your_submission.zip
├── manifest.json      ← required
└── agent.py           ← or whatever you name your entry point
```

**`manifest.json` format:**

```json
{
  "team_name":   "Your Team Name",
  "team_id":     "your_unstop_id",
  "institution": "Your College / Company",
  "members":     ["Name 1", "Name 2", "Name 3"],
  "entry_point": "agent.py"
}
```

- `team_name` and `team_id` must match your Unstop registration.
- `entry_point` is the filename containing your `run_agent` function.
- You may include helper `.py` files. All files are scanned before execution.
- The ZIP may contain subdirectories; `manifest.json` will be found automatically.

---

## Rules

1. Your `run_agent` function must not crash due to `BudgetExceededError` — handle your budget carefully.
2. You may not hardcode information about the test dataset.
3. All code must run within **5 minutes**. Submissions that time out receive whatever score was produced up to the cutoff (which may be 0).
4. No external network calls. No reading/writing files. No spawning processes.
5. One submission per team. The last valid submission before the deadline is evaluated.

---

## Strategy Tips

- **Random sampling gets ~8% fraud in your 100 labels** — the same rate as the overall dataset.
  Smart sampling can yield 40–60% fraud in your labeled set, dramatically improving your classifier.
- **The fraud cases have cluster structure in feature space.** Spending your budget to find these clusters is more valuable than sampling uniformly at random.
- **Class imbalance matters** — with only ~8 positive labels from random sampling, most classifiers will struggle. Consider `class_weight='balanced'` or oversampling.
- **Iterative strategies** — train a model on early labels, use it to decide where to query next.

---

## File Summary

| File | Purpose |
|---|---|
| `dataset.csv` | 10,000-row feature matrix (no labels) |
| `labels.npy` | Local labels for testing — **do not submit** |
| `oracle.py` | Oracle class (budget enforcement) — same interface as evaluation |
| `framework.py` | Local test harness — identical to what we run during grading |
| `example_agent.py` | Baseline agent (random sampling + logistic regression) |
| `requirements.txt` | Python dependencies |
| `README.md` | This file |
