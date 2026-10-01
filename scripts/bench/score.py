"""score.py — score benchmark runs against the exact answer in their task file.

Prerequisites: run directories written by rlm_loop.py, and the task files they
    reference through metadata's `task_file`.
Outputs:       a table on stdout; with --json, one JSON line per run.

Contract: scoring is exact and LLM-free. The predicted and expected payloads
    are canonicalized by the same order-insensitive normalizer, so a correct
    answer never fails on ordering, and a wrong one never passes on it.

Usage:
    uv run python scripts/bench/score.py results/runs/<batch>/<run>
    uv run python scripts/bench/score.py --json results/runs/<batch>/<run>
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from canonical import InvalidAnswer, canonical, canonical_json, parse_answer

ROOT = HERE.parents[1]


def find_task(run_dir: Path, metadata: dict) -> Path | None:
    explicit = metadata.get("task_file")
    if explicit and Path(explicit).exists():
        return Path(explicit)
    task_id = re.sub(r"_\d{8}_\d{6}_\d+$", "", run_dir.name)
    candidates = list((ROOT / "tasks").glob(f"*/{task_id}.json"))
    if len(candidates) == 1:
        return candidates[0]
    return None


def field_diff(expected, got) -> list[str]:
    """Which top-level fields disagree — diagnosis only, never the score."""
    if not isinstance(expected, dict) or not isinstance(got, dict):
        return ["<whole answer>"]
    return sorted(k for k in set(expected) | set(got)
                  if canonical(expected.get(k)) != canonical(got.get(k)))


def score_run(run_dir: Path) -> dict | None:
    meta_path = run_dir / "metadata.json"
    if not meta_path.exists():
        return None
    metadata = json.loads(meta_path.read_text())
    task_path = find_task(run_dir, metadata)
    if task_path is None:
        return None
    task = json.loads(task_path.read_text())
    expected = task["answer"]["expected"]
    text = (run_dir / "answer.md").read_text()

    error, wrong = None, []
    try:
        got = parse_answer(text)
        exact = canonical_json(got) == canonical_json(expected)
        if not exact:
            wrong = field_diff(expected, got)
    except InvalidAnswer as exc:
        exact, error, wrong = False, str(exc), ["<unparsable>"]

    return {
        "run": run_dir.name, "task": task["id"],
        "category": task["taxonomy"]["category"],
        "type": task["taxonomy"]["type"],
        "exact": exact, "wrong_fields": wrong, "format_error": error,
        "reads": metadata.get("charged", metadata.get("reads")),
        "iterations": metadata.get("iterations"),
        "turns": ((metadata.get("generation") or {}).get("turns_override")
                  or task["limits"]["turns"]),
        "shape_rejections": metadata.get("shape_rejections"),
        "status": metadata.get("status"),
        "total_tokens": metadata.get("total_tokens"),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dirs", nargs="+", type=Path)
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    rows = [r for r in (score_run(d) for d in args.run_dirs if d.is_dir())
            if r]
    if args.json:
        for row in rows:
            print(json.dumps(row))
        return

    print(f"{'task':<40} {'ok':<4} {'reads':>7} {'turns':>8} "
          f"{'status':<10} wrong")
    for r in rows:
        print(f"{r['task'][:40]:<40} "
              f"{'yes' if r['exact'] else 'no':<4} "
              f"{str(r['reads']):>7} "
              f"{str(r['iterations']) + '/' + str(r['turns']):>8} "
              f"{str(r['status'])[:10]:<10} {','.join(r['wrong_fields'])[:34]}")
    if rows:
        by = {}
        for r in rows:
            by.setdefault(r["category"], []).append(r)
        print()
        for cat, items in sorted(by.items()):
            ok = sum(i["exact"] for i in items)
            print(f"{cat:<20} {ok}/{len(items)} exact")


if __name__ == "__main__":
    main()
