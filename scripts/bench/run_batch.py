"""run_batch.py — run benchmark tasks once and archive one self-contained batch.

Prerequisites: task JSON files and a reachable endpoint. Export BASE_URL,
    API_KEY, MODEL_NAME and the generation settings first (see README.md).
Outputs:       results/runs/<batch>/ with per-task run directories,
    batch.json, scores.json and failures.json.

Contract: one subprocess per task, so a failed API call costs one task and not
    the batch. Tasks run concurrently on a thread pool — each writes to its own
    `<task_id>_<timestamp>` directory, and the graph is memory-mapped, so the
    processes share pages rather than each loading the graph. A transient
    BadRequestError arriving before any model turn is retried once.

Usage:
    uv run python scripts/bench/run_batch.py --families eval
    uv run python scripts/bench/run_batch.py --families eval --limit 3
    uv run python scripts/bench/run_batch.py --workers 1     # serial
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import datetime as dt
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
LOOP = ROOT / "scripts" / "rlm_loop.py"
SCORE = ROOT / "scripts" / "bench" / "score.py"


def shown(path: Path) -> str:
    """A path relative to the repository when it lies inside it, else absolute."""
    return str(path.relative_to(ROOT) if path.is_relative_to(ROOT) else path)


def discover_tasks(families: list[str], limit: int | None = None) -> list[Path]:
    """Return a stable, duplicate-free task list."""
    tasks: list[Path] = []
    for family in families:
        found = sorted((ROOT / "tasks" / family).glob("*.json"))
        tasks.extend(found[:limit] if limit else found)
    return list(dict.fromkeys(tasks))


def run_one(task: Path, out_root: Path, model: str | None = None,
            retries: int = 2) -> dict:
    command = [sys.executable, str(LOOP), "--task-file", str(task),
               "--output-dir", str(out_root)]
    if model:
        command += ["--model", model]
    dirs: list[Path] = []
    attempts = []
    for attempt in range(1, retries + 1):
        completed = subprocess.run(
            command, capture_output=True, text=True, cwd=ROOT)
        attempts.append({
            "attempt": attempt,
            "returncode": completed.returncode,
            "stdout": completed.stdout[-10_000:],
            "stderr": completed.stderr[-10_000:],
        })
        dirs = sorted(out_root.glob(f"{task.stem}_*"))
        if not dirs:
            continue
        meta = dirs[-1] / "metadata.json"
        if meta.exists() and json.loads(meta.read_text()).get("status") \
                != "crashed:BadRequestError":
            return {"task": task, "run": dirs[-1], "attempts": attempts}
    return {"task": task, "run": dirs[-1] if dirs else None,
            "attempts": attempts}


def git_dir() -> Path | None:
    """This checkout's git directory.

    `.git` is a directory in a clone and a FILE in a linked worktree, holding
    `gitdir: <path>`. Assuming the directory form returns None from a worktree,
    which is where this was first tested and where it first failed.
    """
    candidate = ROOT / ".git"
    if candidate.is_dir():
        return candidate
    if candidate.is_file():
        text = candidate.read_text().strip()
        if text.startswith("gitdir:"):
            return Path(text.split(":", 1)[1].strip())
    return None


def revision_from_git_dir() -> str | None:
    """Resolve HEAD by reading .git, with no git binary.

    A compute node need not have git on PATH, and when it does not, the
    subprocess route archives `git_revision: null`. An archive that cannot be
    tied to the code that produced it is a hole in a corpus meant to be
    released, and the sha is in .git.
    """
    root = git_dir()
    if root is None:
        return None
    head = root / "HEAD"
    if not head.is_file():
        return None
    text = head.read_text().strip()
    if not text.startswith("ref:"):
        return text or None                      # detached HEAD: already a sha
    ref = text.split(None, 1)[1]
    # A linked worktree keeps its own HEAD but shares refs with the common dir.
    common = root
    commondir = root / "commondir"
    if commondir.is_file():
        common = (root / commondir.read_text().strip()).resolve()
    for base in (root, common):
        loose = base / ref
        if loose.is_file():
            return loose.read_text().strip() or None
        packed = base / "packed-refs"             # the ref may only be packed
        if packed.is_file():
            for line in packed.read_text().splitlines():
                if line.startswith("#") or " " not in line:
                    continue
                sha, name = line.split(None, 1)
                if name.strip() == ref:
                    return sha
    return None


def generation_settings() -> dict:
    """Every environment switch that changes what the model does.

    A batch must carry its own configuration:
    temperature, reasoning effort, sub-query budget and harness switches, not
    only the git revision and the worker count.

    Read from the environment rather than from a hardcoded list of defaults,
    so a switch added later is recorded without touching this function --
    anything named RLM_* or SUBQ_* is configuration by construction.
    """
    tracked = {k: v for k, v in os.environ.items()
               if k.startswith(("RLM_", "SUBQ_")) or k == "MAX_NUM_SEQS"}
    return dict(sorted(tracked.items()))


def git_revision() -> str | None:
    """The commit this batch ran, by whichever route works here.

    RLM_GIT_REVISION wins when set, so a job can stamp the revision it was
    submitted at rather than whatever the node can work out.
    """
    stamped = os.environ.get("RLM_GIT_REVISION")
    if stamped:
        return stamped.strip()
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True,
            cwd=ROOT)
    except OSError:
        # No git binary at all: subprocess RAISES rather than returning a
        # non-zero code, so a returncode check never reaches the fallback —
        # exactly the compute-node case this fallback exists for.
        return revision_from_git_dir()
    if completed.returncode == 0 and completed.stdout.strip():
        return completed.stdout.strip()
    return revision_from_git_dir()


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=1, ensure_ascii=False))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--families", nargs="+", default=["eval"])
    ap.add_argument("--task-files", nargs="+", type=Path,
                    help="explicit task files; overrides --families")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--tag", default=None,
                    help="batch name; a YYYYMMDD-HHMM_ prefix is added")
    ap.add_argument("--output-dir", type=Path,
                    help="exact batch directory; overrides --tag")
    ap.add_argument("--model", default=None,
                    help="override MODEL_NAME for this batch")
    # each worker is a subprocess that memory-maps the whole graph, so the
    # default stays conservative on small machines
    ap.add_argument("--workers", type=int, default=2,
                    help="concurrent task subprocesses; 1 runs serially")
    args = ap.parse_args()

    families = args.families

    if args.task_files:
        tasks = [path if path.is_absolute() else ROOT / path
                 for path in args.task_files]
    else:
        tasks = discover_tasks(families, args.limit)
    missing = [str(path) for path in tasks if not path.is_file()]
    if missing:
        ap.error(f"missing task files: {', '.join(missing)}")
    if not tasks:
        ap.error("no task files selected")

    started = dt.datetime.now(dt.UTC)
    if args.output_dir:
        out_root = args.output_dir if args.output_dir.is_absolute() \
            else ROOT / args.output_dir
    else:
        tag = dt.datetime.now().strftime("%Y%m%d-%H%M_") + (args.tag or "batch")
        out_root = ROOT / "results" / "runs" / tag
    out_root.mkdir(parents=True, exist_ok=True)
    print(f"{len(tasks)} tasks -> {out_root}", flush=True)

    manifest = {
        "schema_version": 1,
        "status": "running",
        "started_at": started.isoformat(),
        "completed_at": None,
        "git_revision": git_revision(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "model": args.model or os.environ.get("MODEL_NAME"),
        "families": families if not args.task_files else None,
        "workers": max(1, args.workers),
        "generation": generation_settings(),
        "tasks": [shown(path) for path in tasks],
        "task_count": len(tasks),
        "completed": 0,
        "failed": 0,
    }
    write_json(out_root / "batch.json", manifest)

    done: list[Path] = []
    failures: list[dict] = []
    finished = 0
    with cf.ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        pending = {pool.submit(run_one, task, out_root, args.model): task
                   for task in tasks}
        for future in cf.as_completed(pending):
            finished += 1
            task = pending[future]
            result = future.result()
            run = result["run"]
            print(f"[{finished}/{len(tasks)}] {task.stem}"
                  f"{'' if run else '  FAILED'}", flush=True)
            if run:
                done.append(run)
            else:
                failures.append({
                    "task": shown(task),
                    "attempts": result["attempts"],
                })
            manifest["completed"] = finished
            manifest["failed"] = len(failures)
            write_json(out_root / "batch.json", manifest)
    done.sort()

    scored = subprocess.run(
        [sys.executable, str(SCORE), "--json", *[str(d) for d in done]],
        capture_output=True, text=True, cwd=ROOT)
    rows = [json.loads(line) for line in scored.stdout.splitlines()
            if line.startswith("{")]
    write_json(out_root / "scores.json", rows)
    write_json(out_root / "failures.json", failures)
    manifest["status"] = "completed" if not failures else "completed_with_failures"
    manifest["completed_at"] = dt.datetime.now(dt.UTC).isoformat()
    manifest["wall_time_s"] = round(
        (dt.datetime.now(dt.UTC) - started).total_seconds(), 1)
    manifest["scored"] = len(rows)
    manifest["exact"] = sum(bool(row.get("exact")) for row in rows)
    write_json(out_root / "batch.json", manifest)
    print(f"\nwrote {out_root}/scores.json")
    subprocess.run([sys.executable, str(SCORE), *[str(d) for d in done]],
                   cwd=ROOT)


if __name__ == "__main__":
    main()
