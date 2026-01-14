#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


_RE_JOBFILE = re.compile(r"^hpo-(?P<study>.+)_(?P<ts>\d{8}-\d{6})_(?P<jobid>\d+)_(?P<array>\d+)\.(?P<ext>out|err)$")

_RE_TRIAL_FINISHED = re.compile(r"Trial (?P<trial>\d+) finished\b")
_RE_TRIAL_PRUNED = re.compile(r"Trial (?P<trial>\d+) pruned\b")
_RE_TRIAL_FAILED = re.compile(r"Trial (?P<trial>\d+) failed\b")

# OOM-ish signals seen in this repo's SLURM / subprocess logs
_RE_OOM_LINE = re.compile(r"(oom-kill|out-of-memory|cgroup out-of-memory|Detected \d+ oom-kill event\(s\)|<Signals\.SIGKILL: 9>|Killed\s+\d+\s+Killed)", re.IGNORECASE)
_RE_EXITCODE_LINE = re.compile(r"exit code (?P<code>-?\d+)")
_RE_SIGKILL_LINE = re.compile(r"(SIGKILL|Signals\\.SIGKILL: 9|exit code -9)", re.IGNORECASE)
_RE_HPO_OOM_MARKER = re.compile(r"\[HPO\]\[OOM\]", re.IGNORECASE)

_RE_TRACEBACK = re.compile(r"(Traceback \\(most recent call last\\):|\\bException\\b|\\bError\\b)", re.IGNORECASE)


_STATUS_ORDER = {
    "unknown": 0,
    "finished": 1,
    "pruned": 2,
    "failed": 3,
    "oom": 4,
}


def _merge_status(old: str | None, new: str) -> str:
    if old is None:
        return new
    return new if _STATUS_ORDER[new] > _STATUS_ORDER[old] else old


@dataclass(frozen=True)
class JobKey:
    study: str
    ts: str
    jobid: str

    @property
    def label(self) -> str:
        return f"{self.study}_{self.ts}_{self.jobid}"


def _iter_job_dirs(base: Path) -> Iterable[Path]:
    if base.is_dir() and base.name.startswith("hpo-"):
        yield base
        return
    for p in sorted(base.glob("hpo-*")):
        if p.is_dir():
            yield p


def _job_files(job_dir: Path) -> list[Path]:
    files = []
    for p in job_dir.iterdir():
        if p.is_file() and p.name.endswith((".out", ".err")) and p.name.startswith("hpo-"):
            files.append(p)
    return sorted(files)


def _parse_job_key(job_dir: Path) -> JobKey:
    # Prefer parsing from filenames (authoritative jobid).
    for f in _job_files(job_dir):
        m = _RE_JOBFILE.match(f.name)
        if m:
            return JobKey(study=m.group("study"), ts=m.group("ts"), jobid=m.group("jobid"))
    # Fallback: parse from directory name; jobid may be unknown.
    name = job_dir.name
    if name.startswith("hpo-"):
        name = name[len("hpo-") :]
    parts = name.rsplit("_", 1)
    if len(parts) == 2 and re.fullmatch(r"\d{8}-\d{6}", parts[1]):
        return JobKey(study=parts[0], ts=parts[1], jobid="unknown")
    return JobKey(study=name, ts="unknown", jobid="unknown")


def _classify_file(text: str) -> tuple[dict[int, str], bool, bool]:
    """
    Returns:
      trial_status: trial_id -> status (finished/pruned/failed/oom)
      saw_oom: True if any OOM marker present in the file
      saw_any_trial: True if any Trial <id> status line present
    """
    trial_status: dict[int, str] = {}
    last_trial: int | None = None
    saw_any_trial = False
    saw_oom = False

    for line in text.splitlines():
        m = _RE_TRIAL_FINISHED.search(line)
        if m:
            tid = int(m.group("trial"))
            saw_any_trial = True
            last_trial = tid
            trial_status[tid] = _merge_status(trial_status.get(tid), "finished")
            continue

        m = _RE_TRIAL_PRUNED.search(line)
        if m:
            tid = int(m.group("trial"))
            saw_any_trial = True
            last_trial = tid
            trial_status[tid] = _merge_status(trial_status.get(tid), "pruned")
            continue

        m = _RE_TRIAL_FAILED.search(line)
        if m:
            tid = int(m.group("trial"))
            saw_any_trial = True
            last_trial = tid
            trial_status[tid] = _merge_status(trial_status.get(tid), "failed")
            # If the failure line itself suggests SIGKILL/-9, upgrade to OOM.
            if _RE_SIGKILL_LINE.search(line):
                trial_status[tid] = _merge_status(trial_status.get(tid), "oom")
            continue

        if _RE_HPO_OOM_MARKER.search(line) or _RE_OOM_LINE.search(line):
            saw_oom = True
            # Most OOM lines don't carry trial id; attribute to last seen trial if available.
            if last_trial is not None:
                trial_status[last_trial] = _merge_status(trial_status.get(last_trial), "oom")
            continue

        # Some logs include "exit code -9" on separate lines (subprocess wrapper).
        if "exit code" in line.lower():
            mcode = _RE_EXITCODE_LINE.search(line.lower())
            if mcode and mcode.group("code") == "-9":
                saw_oom = True
                if last_trial is not None:
                    trial_status[last_trial] = _merge_status(trial_status.get(last_trial), "oom")

    # If file had OOM markers but no last_trial, upgrade all failed trials in file.
    if saw_oom and last_trial is None:
        for tid, st in list(trial_status.items()):
            if st in {"failed"}:
                trial_status[tid] = _merge_status(st, "oom")

    return trial_status, saw_oom, saw_any_trial


def _summarize_job(job_dir: Path) -> dict:
    key = _parse_job_key(job_dir)
    files = _job_files(job_dir)

    merged_trials: dict[int, str] = {}
    empty_workers = 0
    oom_workers = 0
    failed_pretrial_workers = 0
    worker_status_counts = {k: 0 for k in ["finished", "pruned", "failed", "oom", "empty", "failed_pretrial"]}

    # Group by (jobid,arrayid) = worker
    by_worker: dict[tuple[str, str], list[Path]] = {}
    for f in files:
        m = _RE_JOBFILE.match(f.name)
        if not m:
            continue
        by_worker.setdefault((m.group("jobid"), m.group("array")), []).append(f)

    for (_jobid, _arr), flist in sorted(by_worker.items(), key=lambda x: (int(x[0][1]), x[0][0])):
        worker_trials: dict[int, str] = {}
        saw_any = False
        worker_oom = False
        worker_text = ""
        for f in sorted(flist, key=lambda p: p.suffix):
            try:
                text = f.read_text(errors="replace")
            except Exception:
                continue
            worker_text += "\n" + text
            tmap, saw_oom, saw_trial = _classify_file(text)
            saw_any = saw_any or saw_trial
            worker_oom = worker_oom or saw_oom
            for tid, st in tmap.items():
                worker_trials[tid] = _merge_status(worker_trials.get(tid), st)

        if worker_oom:
            oom_workers += 1

        # Worker-level status: in this repo each array task typically runs exactly one trial.
        if not saw_any and not worker_trials:
            # If we have an obvious traceback/error but no Optuna "Trial X ..." line, treat it as pre-trial failure.
            if _RE_TRACEBACK.search(worker_text):
                failed_pretrial_workers += 1
                worker_status_counts["failed_pretrial"] += 1
            else:
                empty_workers += 1
                worker_status_counts["empty"] += 1
        else:
            # Pick the "worst" status observed for that worker.
            worst = "finished"
            for st in worker_trials.values():
                worst = _merge_status(worst, st)
            worker_status_counts[worst] = worker_status_counts.get(worst, 0) + 1

        for tid, st in worker_trials.items():
            merged_trials[tid] = _merge_status(merged_trials.get(tid), st)

    counts = {k: 0 for k in _STATUS_ORDER.keys() if k != "unknown"}
    for st in merged_trials.values():
        counts[st] = counts.get(st, 0) + 1

    total = len(merged_trials)
    finished = counts.get("finished", 0)
    pruned = counts.get("pruned", 0)
    failed = counts.get("failed", 0)
    oom = counts.get("oom", 0)

    unsuccessful = total - finished
    workers_total = len(by_worker)
    workers_finished = worker_status_counts.get("finished", 0)
    workers_pruned = worker_status_counts.get("pruned", 0)
    workers_failed = worker_status_counts.get("failed", 0) + worker_status_counts.get("failed_pretrial", 0)
    workers_oom = worker_status_counts.get("oom", 0)
    workers_empty = worker_status_counts.get("empty", 0)
    workers_unsuccessful = workers_total - workers_finished

    # Try to locate corresponding Optuna artifacts for this HPO job root.
    # Convention in this repo: outputs/optuna/<study>_<timestamp>/trials/trial_XXXXX/
    optuna_trials_total = 0
    optuna_metrics_json = 0
    optuna_failure_json = 0
    optuna_root = Path("outputs/optuna") / f"{key.study}_{key.ts}"
    if optuna_root.exists():
        trial_dirs = sorted([p for p in (optuna_root / "trials").glob("trial_*") if p.is_dir()])
        optuna_trials_total = len(trial_dirs)
        for td in trial_dirs:
            if (td / "metrics.json").exists():
                optuna_metrics_json += 1
            if (td / "failure.json").exists():
                optuna_failure_json += 1

    return {
        "job_dir": str(job_dir),
        "study": key.study,
        "timestamp": key.ts,
        "jobid": key.jobid,
        "total_trials": total,
        "finished": finished,
        "pruned": pruned,
        "failed": failed,
        "oom": oom,
        "unsuccessful": unsuccessful,
        "unsuccessful_rate": (unsuccessful / total) if total else float("nan"),
        "workers_total": workers_total,
        "workers_finished": workers_finished,
        "workers_pruned": workers_pruned,
        "workers_failed": workers_failed,
        "workers_oom": workers_oom,
        "workers_empty": workers_empty,
        "workers_unsuccessful_rate": (workers_unsuccessful / workers_total) if workers_total else float("nan"),
        "empty_workers": empty_workers,
        "oom_workers": oom_workers,
        "failed_pretrial_workers": failed_pretrial_workers,
        "optuna_root": str(optuna_root) if optuna_root.exists() else "",
        "optuna_trials_total": optuna_trials_total,
        "optuna_metrics_json": optuna_metrics_json,
        "optuna_failure_json": optuna_failure_json,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Compute per-HPO-job trial outcome stats from logs/hpo folders.")
    ap.add_argument(
        "--logs-root",
        default="logs/hpo",
        help="Root directory containing hpo-* folders (default: logs/hpo). You can also pass a single hpo-* folder.",
    )
    ap.add_argument(
        "--jobs",
        nargs="*",
        default=None,
        help="Optional: one or more hpo-* folder names to include (basename match). If omitted, analyzes all under --logs-root.",
    )
    args = ap.parse_args()

    base = Path(args.logs_root)
    if not base.exists():
        raise SystemExit(f"logs root not found: {base}")

    job_dirs = list(_iter_job_dirs(base))
    if args.jobs:
        allowed = set(args.jobs)
        job_dirs = [d for d in job_dirs if d.name in allowed]

    if not job_dirs:
        raise SystemExit(f"No hpo-* job dirs found under: {base}")

    rows = [_summarize_job(d) for d in job_dirs]
    # stable sort: study, ts, jobid
    def _sort_key(r: dict):
        jid = r["jobid"]
        try:
            jid_int = int(jid)
        except Exception:
            jid_int = 1_000_000_000
        return (r["study"], r["timestamp"], jid_int)

    rows = sorted(rows, key=_sort_key)

    header = [
        "job_dir",
        "jobid",
        "total_trials",
        "finished",
        "pruned",
        "failed",
        "oom",
        "unsuccessful_rate",
        "workers_total",
        "workers_unsuccessful_rate",
        "workers_empty",
        "workers_oom",
        "failed_pretrial_workers",
        "optuna_trials_total",
        "optuna_metrics_json",
        "optuna_failure_json",
    ]
    print("\t".join(header))
    for r in rows:
        print(
            "\t".join(
                [
                    Path(r["job_dir"]).name,
                    str(r["jobid"]),
                    str(r["total_trials"]),
                    str(r["finished"]),
                    str(r["pruned"]),
                    str(r["failed"]),
                    str(r["oom"]),
                    f"{r['unsuccessful_rate']:.3f}" if r["total_trials"] else "nan",
                    str(r["workers_total"]),
                    f"{r['workers_unsuccessful_rate']:.3f}" if r["workers_total"] else "nan",
                    str(r["workers_empty"]),
                    str(r["workers_oom"]),
                    str(r["failed_pretrial_workers"]),
                    str(r["optuna_trials_total"]),
                    str(r["optuna_metrics_json"]),
                    str(r["optuna_failure_json"]),
                ]
            )
        )


if __name__ == "__main__":
    main()
