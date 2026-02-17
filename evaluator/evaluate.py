"""
Main evaluator — processes all submission ZIPs and produces results.csv.

Usage (from this directory):
    python evaluate.py \\
        --submissions-dir submissions/ \\
        --eval-features   ../eval_dataset/eval_features.csv \\
        --eval-labels     ../eval_dataset/eval_labels.npy \\
        --output          results.csv \\
        --workers         10

Drop all team ZIP files in submissions/ then run the command above.
Results stream to results.csv as they complete — safe to Ctrl-C and resume
(already-written rows are preserved; re-run overwrites the file entirely).
"""

import csv
import json
import sys
import time
import signal
import zipfile
import tempfile
import traceback
import importlib.util
from pathlib import Path
from datetime import datetime, timezone
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score

# Local modules — must be in the same directory.
sys.path.insert(0, str(Path(__file__).parent))
from ast_check import check_submission
from oracle import Oracle, BudgetExceededError

# ─── Constants ────────────────────────────────────────────────────────────────
BUDGET          = 100
TIMEOUT_SECONDS = 300        # 5 minutes per submission
MANIFEST_FILE   = "manifest.json"

RESULT_FIELDS = [
    "rank",
    "team_name",
    "team_id",
    "institution",
    "zip_filename",
    "f1_score",
    "precision",
    "recall",
    "queries_used",
    "runtime_seconds",
    "status",
    "error_message",
    "evaluated_at",
]


# ─── Timeout (Unix only — RunPod is Linux) ────────────────────────────────────

class _TimeoutError(Exception):
    pass


def _alarm_handler(signum, frame):
    raise _TimeoutError(f"Agent exceeded {TIMEOUT_SECONDS}s time limit")


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _load_agent(entry_point: Path):
    spec = importlib.util.spec_from_file_location("_agent_module", entry_point)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _coerce_predictions(raw, n: int) -> np.ndarray:
    """
    Accept any array-like and return a clean int array of length n.

    - None / missing → all zeros (F1 = 0).
    - Shorter than n  → padded with 0 (majority-class assumption).
    - Longer than n   → truncated.
    - Float probabilities → thresholded at 0.5.
    """
    if raw is None:
        return np.zeros(n, dtype=int)
    try:
        arr = np.asarray(raw, dtype=float).flatten()
    except Exception:
        return np.zeros(n, dtype=int)

    if len(arr) < n:
        arr = np.concatenate([arr, np.zeros(n - len(arr))])
    elif len(arr) > n:
        arr = arr[:n]

    return (arr >= 0.5).astype(int)


def _score(labels: np.ndarray, preds: np.ndarray):
    f1   = float(f1_score(labels,   preds, zero_division=0))
    prec = float(precision_score(labels, preds, zero_division=0))
    rec  = float(recall_score(labels,  preds, zero_division=0))
    return round(f1, 6), round(prec, 6), round(rec, 6)


# ─── Per-submission worker (runs in a child process) ─────────────────────────

def _evaluate_one(zip_path: Path, eval_df: pd.DataFrame, eval_labels: np.ndarray) -> dict:
    """
    Evaluate a single submission ZIP.
    Always returns a result dict — never raises.
    """
    n = len(eval_df)

    result = {
        "rank":           None,
        "team_name":      "UNKNOWN",
        "team_id":        "UNKNOWN",
        "institution":    "UNKNOWN",
        "zip_filename":   zip_path.name,
        "f1_score":       0.0,
        "precision":      0.0,
        "recall":         0.0,
        "queries_used":   0,
        "runtime_seconds": 0.0,
        "status":         "ERROR",
        "error_message":  "",
        "evaluated_at":   datetime.now(timezone.utc).isoformat(),
    }

    with tempfile.TemporaryDirectory() as tmpdir:
        team_dir = Path(tmpdir)

        # ── 1. Extract ZIP ────────────────────────────────────────────────────
        try:
            with zipfile.ZipFile(zip_path) as zf:
                # Safety: reject absolute paths and path traversal
                for member in zf.namelist():
                    if member.startswith("/") or ".." in member:
                        result["error_message"] = f"Unsafe path in ZIP: {member}"
                        return result
                zf.extractall(team_dir)
        except zipfile.BadZipFile as e:
            result["error_message"] = f"Bad ZIP file: {e}"
            return result
        except Exception as e:
            result["error_message"] = f"ZIP extraction error: {e}"
            return result

        # ── 2. Find manifest.json ─────────────────────────────────────────────
        manifests = list(team_dir.rglob(MANIFEST_FILE))
        if not manifests:
            result["error_message"] = f"'{MANIFEST_FILE}' not found in submission"
            return result

        manifest_path = manifests[0]
        submission_root = manifest_path.parent

        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception as e:
            result["error_message"] = f"Cannot parse {MANIFEST_FILE}: {e}"
            return result

        result["team_name"]   = str(manifest.get("team_name",   "UNKNOWN"))
        result["team_id"]     = str(manifest.get("team_id",     "UNKNOWN"))
        result["institution"] = str(manifest.get("institution", "UNKNOWN"))
        entry_point_name      = str(manifest.get("entry_point", "agent.py"))

        entry_point = submission_root / entry_point_name
        if not entry_point.exists():
            result["error_message"] = (
                f"Entry point '{entry_point_name}' declared in manifest not found"
            )
            return result

        # ── 3. AST security scan ──────────────────────────────────────────────
        violations = check_submission(submission_root)
        if violations:
            result["status"] = "DISQUALIFIED"
            # Show first 3 violations max to keep CSV readable
            result["error_message"] = " | ".join(violations[:3])
            if len(violations) > 3:
                result["error_message"] += f" | ... ({len(violations)} total)"
            return result

        # ── 4. Load and run agent ─────────────────────────────────────────────
        oracle = Oracle(eval_labels, budget=BUDGET)
        start_time = time.perf_counter()
        raw_preds = None

        try:
            signal.signal(signal.SIGALRM, _alarm_handler)
            signal.alarm(TIMEOUT_SECONDS)
            try:
                agent = _load_agent(entry_point)

                if not hasattr(agent, "run_agent"):
                    result["error_message"] = (
                        "'run_agent' function not found in entry point"
                    )
                    return result

                raw_preds = agent.run_agent(eval_df.copy(), oracle, BUDGET)

            finally:
                signal.alarm(0)  # cancel alarm

        except _TimeoutError as e:
            result["status"] = "TIMEOUT"
            result["error_message"] = str(e)
            result["queries_used"] = oracle.queries_used
            result["runtime_seconds"] = round(time.perf_counter() - start_time, 2)
            # Still score whatever was returned (raw_preds may be None)
            preds = _coerce_predictions(raw_preds, n)
            result["f1_score"], result["precision"], result["recall"] = _score(eval_labels, preds)
            return result

        except BudgetExceededError as e:
            result["status"] = "BUDGET_EXCEEDED"
            result["error_message"] = str(e)
            result["queries_used"] = oracle.queries_used
            result["runtime_seconds"] = round(time.perf_counter() - start_time, 2)
            preds = _coerce_predictions(raw_preds, n)
            result["f1_score"], result["precision"], result["recall"] = _score(eval_labels, preds)
            return result

        except Exception as e:
            result["status"] = "ERROR"
            result["error_message"] = f"{type(e).__name__}: {e}"
            result["queries_used"] = oracle.queries_used
            result["runtime_seconds"] = round(time.perf_counter() - start_time, 2)
            preds = _coerce_predictions(raw_preds, n)
            result["f1_score"], result["precision"], result["recall"] = _score(eval_labels, preds)
            return result

        elapsed = round(time.perf_counter() - start_time, 2)

        # ── 5. Score ──────────────────────────────────────────────────────────
        preds = _coerce_predictions(raw_preds, n)
        f1, prec, rec = _score(eval_labels, preds)

        result.update({
            "f1_score":        f1,
            "precision":       prec,
            "recall":          rec,
            "queries_used":    oracle.queries_used,
            "runtime_seconds": elapsed,
            "status":          "OK",
            "error_message":   "",
        })

    return result


# ─── Ranking ─────────────────────────────────────────────────────────────────

def _assign_ranks(rows: list[dict]) -> list[dict]:
    """
    Sort and assign ranks to OK submissions.

    Tie-breaking order (all ascending unless noted):
      1. f1_score          (higher is better → descending)
      2. queries_used      (fewer is better → ascending)
      3. runtime_seconds   (faster is better → ascending)

    Non-OK submissions have rank = None.
    """
    ok = [r for r in rows if r["status"] == "OK"]
    ok.sort(key=lambda r: (
        -r["f1_score"],
        r["queries_used"],
        r["runtime_seconds"],
    ))
    for i, r in enumerate(ok, start=1):
        r["rank"] = i

    # Preserve original order for non-OK
    rank_map = {r["zip_filename"]: r["rank"] for r in ok}
    for r in rows:
        r["rank"] = rank_map.get(r["zip_filename"])
    return rows


# ─── Main evaluation loop ─────────────────────────────────────────────────────

def run_evaluation(
    submissions_dir: Path,
    eval_features_path: Path,
    eval_labels_path: Path,
    output_csv: Path,
    max_workers: int = 10,
) -> None:
    eval_df     = pd.read_csv(eval_features_path)
    eval_labels = np.load(eval_labels_path)

    zip_files = sorted(submissions_dir.glob("*.zip"))
    if not zip_files:
        print(f"No ZIP files found in {submissions_dir}")
        return

    print(f"Found {len(zip_files)} submissions | workers={max_workers}")
    print(f"Eval dataset : {len(eval_df)} rows | "
          f"{eval_labels.sum()} fraud ({eval_labels.mean():.1%})\n")

    all_results: list[dict] = []

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_map = {
            executor.submit(_evaluate_one, zp, eval_df, eval_labels): zp
            for zp in zip_files
        }

        done = 0
        total = len(zip_files)
        for future in as_completed(future_map):
            zp = future_map[future]
            try:
                result = future.result()
            except Exception as e:
                result = {
                    "rank": None,
                    "team_name": "UNKNOWN",
                    "team_id": "UNKNOWN",
                    "institution": "UNKNOWN",
                    "zip_filename": zp.name,
                    "f1_score": 0.0,
                    "precision": 0.0,
                    "recall": 0.0,
                    "queries_used": 0,
                    "runtime_seconds": 0.0,
                    "status": "INTERNAL_ERROR",
                    "error_message": str(e),
                    "evaluated_at": datetime.now(timezone.utc).isoformat(),
                }

            all_results.append(result)
            done += 1

            status_icon = {
                "OK":             "✓",
                "ERROR":          "✗",
                "TIMEOUT":        "⏱",
                "BUDGET_EXCEEDED":"$",
                "DISQUALIFIED":   "⚠",
                "INTERNAL_ERROR": "!",
            }.get(result["status"], "?")

            print(
                f"[{done:>4}/{total}] {status_icon} {result['zip_filename']:<40} "
                f"F1={result['f1_score']:.4f}  "
                f"Q={result['queries_used']:>3}  "
                f"{result['runtime_seconds']:>6.1f}s  "
                f"{result['status']}"
                + (f"  — {result['error_message'][:60]}" if result["error_message"] else "")
            )

    # Assign ranks after all submissions are scored
    all_results = _assign_ranks(all_results)

    # Write CSV (sorted: OK first by rank, then non-OK)
    all_results.sort(key=lambda r: (r["rank"] is None, r["rank"] or 0))

    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=RESULT_FIELDS)
        writer.writeheader()
        writer.writerows(all_results)

    print(f"\nResults saved → {output_csv}")

    # Quick summary
    ok_count    = sum(1 for r in all_results if r["status"] == "OK")
    error_count = sum(1 for r in all_results if r["status"] not in ("OK", "DISQUALIFIED"))
    disq_count  = sum(1 for r in all_results if r["status"] == "DISQUALIFIED")

    print(f"\nSummary: {ok_count} OK | {error_count} errors | {disq_count} disqualified")

    if ok_count:
        ok = [r for r in all_results if r["status"] == "OK"]
        print(f"\n{'Rank':<6}{'Team':<30}{'F1':>8}{'Queries':>9}{'Time':>8}")
        print("-" * 62)
        for r in ok[:20]:
            print(
                f"{r['rank']:<6}{r['team_name']:<30}"
                f"{r['f1_score']:>8.4f}{r['queries_used']:>9}{r['runtime_seconds']:>7.1f}s"
            )
        if len(ok) > 20:
            print(f"  ... and {len(ok) - 20} more. See {output_csv} for full results.")


# ─── CLI ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate hackathon submissions.")
    parser.add_argument("--submissions-dir", default="submissions/",
                        help="Directory containing team ZIP files")
    parser.add_argument("--eval-features",   default="../eval_dataset/eval_features.csv")
    parser.add_argument("--eval-labels",     default="../eval_dataset/eval_labels.npy")
    parser.add_argument("--output",          default="results.csv")
    parser.add_argument("--workers",         type=int, default=10,
                        help="Parallel worker processes (default: 10)")
    args = parser.parse_args()

    run_evaluation(
        submissions_dir    = Path(args.submissions_dir),
        eval_features_path = Path(args.eval_features),
        eval_labels_path   = Path(args.eval_labels),
        output_csv         = Path(args.output),
        max_workers        = args.workers,
    )
