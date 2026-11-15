# Innov8 4.0: Talent Fraud Detection with a Limited Label Budget

An **active-learning hackathon challenge** built for Innov8 4.0 at IIT Delhi, sponsored by Eightfold AI. Participants get 10,000 unlabelled candidate profiles and may query the ground-truth label for at most **100 rows**. How they spend that budget decides how well their fraud classifier does.

This repo holds the full challenge kit:

| Folder | What it is |
|---|---|
| [`participant_kit/`](participant_kit/) | Problem statement, dataset (`dataset.csv`), label oracle, framework, example agent and a sample submission |
| [`dataset_gen/`](dataset_gen/) | Synthetic data generator: four fraud types (credential fraud, application bombing, account takeover, ghost profiles), each shifting only 3–4 features so that random sampling fails and smart sampling wins |
| [`evaluator/`](evaluator/) | Batch evaluator for submission ZIPs (sandboxed import, AST checks, oracle budget enforcement) and leaderboard generator |
| `eval_dataset/` | Held-out evaluation split used for final scoring |

## Why it is interesting

- **Budgeted oracle**: every label costs one unit, so the problem is about *which* rows to label (uncertainty sampling, clustering, diversity), not only about the model.
- **Designed difficulty**: per-feature AUC stays ≤ 0.72 while within-cluster signals are strong, so about 2 labels per fraud cluster gives F1 ≈ 0.2–0.35 and about 10 per cluster gives F1 ≈ 0.65–0.78.
- **Fully synthetic**: no real candidate data is used anywhere.

## Quick start

```bash
cd participant_kit
pip install -r requirements.txt
python example_agent.py          # baseline using the label oracle
```

Regenerate data:

```bash
cd dataset_gen
python generate.py --split participant --seed 42 --output-dir ../participant_kit
python generate.py --split eval        --seed 99 --output-dir ../eval_dataset
```

Score submissions:

```bash
cd evaluator
pip install -r requirements.txt
python evaluate.py --submissions-dir submissions/ --eval-features ../eval_dataset/eval_features.csv --eval-labels ../eval_dataset/eval_labels.npy --output results.csv
python generate_leaderboard.py
```

See [`participant_kit/README.md`](participant_kit/README.md) for the full problem statement and submission format.
