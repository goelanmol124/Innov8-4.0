"""
Generate a readable leaderboard from results.csv.

Usage:
    python generate_leaderboard.py results.csv
    python generate_leaderboard.py results.csv --format markdown
    python generate_leaderboard.py results.csv --format csv --output leaderboard.csv
"""

import sys
import csv
import argparse
from pathlib import Path


def load_results(results_csv: Path) -> list[dict]:
    with open(results_csv, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _coerce(row: dict) -> dict:
    for float_col in ("f1_score", "precision", "recall", "runtime_seconds"):
        try:
            row[float_col] = float(row[float_col])
        except (ValueError, KeyError):
            row[float_col] = 0.0
    for int_col in ("queries_used",):
        try:
            row[int_col] = int(row[int_col])
        except (ValueError, KeyError):
            row[int_col] = 0
    try:
        row["rank"] = int(row["rank"]) if row.get("rank") else None
    except ValueError:
        row["rank"] = None
    return row


def generate(results_csv: Path, fmt: str = "table", output: Path = None) -> None:
    rows = [_coerce(r) for r in load_results(results_csv)]

    ok   = [r for r in rows if r["status"] == "OK"]
    errs = [r for r in rows if r["status"] not in ("OK", "DISQUALIFIED")]
    disq = [r for r in rows if r["status"] == "DISQUALIFIED"]

    # Sort OK by rank (already assigned by evaluate.py)
    # Tie-break order (mirrors evaluate.py):
    #   1. F1 score      (higher → better)
    #   2. queries_used  (fewer  → better)
    #   3. runtime_s     (faster → better)
    ok.sort(key=lambda r: (
        r["rank"] if r["rank"] is not None else 99999,
    ))

    if fmt == "table":
        _print_table(ok, errs, disq, rows)
    elif fmt == "markdown":
        _print_markdown(ok, errs, disq, rows)
    elif fmt == "csv":
        _write_csv(ok, output or Path("leaderboard.csv"))


# ─── Table format ─────────────────────────────────────────────────────────────

def _print_table(ok, errs, disq, all_rows):
    W_TEAM = 28
    print()
    print("=" * 80)
    print("  LEADERBOARD")
    print("  Tie-breaking: F1 (↑)  →  Queries used (↓)  →  Runtime (↓)")
    print("=" * 80)
    hdr = (
        f"{'Rank':<6}{'Team':<{W_TEAM}}{'Institution':<20}"
        f"{'F1':>8}{'Prec':>8}{'Recall':>8}{'Queries':>9}{'Time(s)':>9}"
    )
    print(hdr)
    print("-" * 80)

    for r in ok:
        print(
            f"{r['rank']:<6}"
            f"{r['team_name'][:W_TEAM-1]:<{W_TEAM}}"
            f"{r['institution'][:19]:<20}"
            f"{r['f1_score']:>8.4f}"
            f"{r['precision']:>8.4f}"
            f"{r['recall']:>8.4f}"
            f"{r['queries_used']:>9}"
            f"{r['runtime_seconds']:>9.1f}"
        )

    print("=" * 80)
    print(f"  Total submissions : {len(all_rows)}")
    print(f"  Scored (OK)       : {len(ok)}")
    print(f"  Errors / partial  : {len(errs)}")
    print(f"  Disqualified      : {len(disq)}")

    if ok:
        top = ok[0]
        print(f"\n  Winner : {top['team_name']} ({top['institution']})  "
              f"F1={top['f1_score']:.4f}  Queries={top['queries_used']}")

    if disq:
        print(f"\n  Disqualified teams:")
        for r in disq:
            reason = r["error_message"][:70]
            print(f"    {r['team_name'][:25]:<25} — {reason}")

    if errs:
        print(f"\n  Teams with errors (scored as 0):")
        for r in errs:
            msg = r["error_message"][:60]
            print(f"    {r['team_name'][:25]:<25} [{r['status']}] {msg}")

    print()


# ─── Markdown format ─────────────────────────────────────────────────────────

def _print_markdown(ok, errs, disq, all_rows):
    lines = [
        "# Leaderboard",
        "",
        "_Tie-breaking: F1 (↑) → Queries used (↓) → Runtime (↓)_",
        "",
        "| Rank | Team | Institution | F1 | Precision | Recall | Queries | Time (s) |",
        "|------|------|-------------|---:|----------:|-------:|--------:|---------:|",
    ]
    for r in ok:
        lines.append(
            f"| {r['rank']} | {r['team_name']} | {r['institution']} "
            f"| {r['f1_score']:.4f} | {r['precision']:.4f} | {r['recall']:.4f} "
            f"| {r['queries_used']} | {r['runtime_seconds']:.1f} |"
        )
    lines += [
        "",
        f"**Total submissions:** {len(all_rows)}  ",
        f"**Scored (OK):** {len(ok)}  ",
        f"**Errors:** {len(errs)}  ",
        f"**Disqualified:** {len(disq)}  ",
        "",
    ]
    if disq:
        lines += ["## Disqualified", ""]
        for r in disq:
            lines.append(f"- **{r['team_name']}**: {r['error_message'][:100]}")
        lines.append("")
    print("\n".join(lines))


# ─── CSV format ──────────────────────────────────────────────────────────────

def _write_csv(ok, output: Path):
    fields = [
        "rank", "team_name", "team_id", "institution",
        "f1_score", "precision", "recall",
        "queries_used", "runtime_seconds",
    ]
    with open(output, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(ok)
    print(f"Leaderboard CSV written → {output}")


# ─── CLI ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("results_csv", nargs="?", default="results.csv")
    parser.add_argument("--format", choices=["table", "markdown", "csv"], default="table")
    parser.add_argument("--output", default=None, help="Output path (for --format csv)")
    args = parser.parse_args()

    generate(
        Path(args.results_csv),
        fmt=args.format,
        output=Path(args.output) if args.output else None,
    )
