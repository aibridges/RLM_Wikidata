"""summary.py — one results table for a batch, overall and per category.

Prerequisites: a batch directory written by run_batch.py, holding scores.json.
Outputs:       a table on stdout; with --json, one JSON object.

Contract: every number is read from scores.json, which score.py writes from
    the archived runs. Nothing is re-scored here. Tasks are grouped by
    category, read from the task id: SE for single-entity, MH for multi-hop.

Usage:
    python scripts/bench/summary.py results/runs/<batch>
    python scripts/bench/summary.py --json results/runs/<batch>
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

PATTERNS = ("SE", "MH")


def pattern(task_id: str) -> str:
    parts = str(task_id).split("_")
    return parts[1] if len(parts) > 1 else parts[0]


def totals(rows: list[dict]) -> dict:
    return {
        "tasks": len(rows),
        "exact": sum(bool(r.get("exact")) for r in rows),
        "reached_final": sum(r.get("status") == "final" for r in rows),
        "max_iters": sum(r.get("status") == "max_iters" for r in rows),
        "turns": sum(int(r.get("iterations") or 0) for r in rows),
        "graph_reads": sum(int(r.get("reads") or 0) for r in rows),
        "total_tokens": sum(int(r.get("total_tokens") or 0) for r in rows),
    }


def summarize(batch: Path) -> dict:
    rows = json.loads((batch / "scores.json").read_text())
    by_pattern = {}
    for name in PATTERNS:
        subset = [r for r in rows if pattern(r["task"]) == name]
        if subset:
            by_pattern[name] = totals(subset)
    manifest = batch / "batch.json"
    meta = json.loads(manifest.read_text()) if manifest.exists() else {}
    return {"batch": batch.name, "model": meta.get("model"),
            "wall_time_s": meta.get("wall_time_s"),
            "generation": meta.get("generation"),
            "overall": totals(rows), "by_pattern": by_pattern}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("batch", type=Path)
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    report = summarize(args.batch)
    if args.json:
        print(json.dumps(report, indent=1, ensure_ascii=False))
        return

    overall = report["overall"]
    n = overall["tasks"] or 1
    print(f"batch  {report['batch']}")
    print(f"model  {report['model']}")
    if report["wall_time_s"]:
        print(f"wall   {report['wall_time_s'] / 60:.1f} min")
    print()
    print(f"exact           {overall['exact']}/{overall['tasks']} "
          f"({100 * overall['exact'] / n:.1f}%)")
    print(f"reached FINAL   {overall['reached_final']}/{overall['tasks']}")
    print(f"max_iters       {overall['max_iters']}")
    print(f"turns           {overall['turns']:,}")
    print(f"graph reads     {overall['graph_reads']:,}")
    print(f"total tokens    {overall['total_tokens']:,}")
    print()
    print(f"{'category':<8}{'tasks':>6}{'exact':>7}"
          f"{'final':>7}{'turns':>7}{'reads':>8}")
    for name, t in report["by_pattern"].items():
        print(f"{name:<8}{t['tasks']:>6}{t['exact']:>7}"
              f"{t['reached_final']:>7}{t['turns']:>7}{t['graph_reads']:>8}")


if __name__ == "__main__":
    main()
