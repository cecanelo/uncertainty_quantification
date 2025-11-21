#!/usr/bin/env python3
"""
Analyze an HPO job array after completion.

Inputs (one of):
  --job-id  <SLURM_JOBID>   # preferred; resolves the job root automatically
  --job-root <PATH>         # alternative: give the job root directly

What it prints:
  • Job root path
  • Trials counted / missing time info
  • Array elapsed time (HH:MM:SS)  = min(start) → max(end) across trials
  • Trial time stats in seconds: min / max / mean / std

Assumptions:
  • Each trial directory lives under: <JOB_ROOT>/trials/trial_000XX/
  • Primary source: trial_*/run_meta.json (written by your train.py) with keys:
        start_utc, end_utc, train_time_sec
  • Fallbacks if missing:
        - end_utc − start_utc if both present
        - newest(metrics.json, model.pt, trial dir mtime) − trial dir mtime
"""

import argparse
import json
import os
from pathlib import Path
from typing import List, Optional, Tuple
from datetime import datetime, timezone, timedelta
import statistics as stats
import math

# -------------------- helpers --------------------

def _to_dt(s: Optional[str]) -> Optional[datetime]:
    if not s:
        return None
    try:
        # Handles ISO like "2025-11-08T13:06:07+00:00"
        return datetime.fromisoformat(s)
    except Exception:
        return None

def _resolve_job_root_by_jobid(job_id: str) -> Optional[Path]:
    base = Path("outputs/optuna")

    # 1) Prefer the symlink pattern created by the updated runner: *_{JOBID}
    candidates = sorted(base.glob(f"*_{job_id}"), key=lambda p: p.stat().st_mtime, reverse=True)
    if candidates:
        return candidates[0].resolve()

    # 2) Fall back to scanning job roots for jobid.txt match
    for d in sorted((p for p in base.iterdir() if p.is_dir()), key=lambda p: p.stat().st_mtime, reverse=True):
        jid = (d / "jobid.txt")
        if jid.exists():
            try:
                if jid.read_text().strip() == job_id:
                    return d.resolve()
            except Exception:
                pass

    # 3) Nothing found
    return None

def _trial_time_from_run_meta(run_meta: dict) -> Tuple[Optional[float], Optional[datetime], Optional[datetime]]:
    """Return (train_time_sec, start_dt, end_dt) from train.py's run_meta.json."""
    tsec = run_meta.get("train_time_sec")
    if isinstance(tsec, (int, float)) and tsec > 0:
        t_val = float(tsec)
    else:
        t_val = None
    start_dt = _to_dt(run_meta.get("start_utc"))
    end_dt   = _to_dt(run_meta.get("end_utc"))
    return t_val, start_dt, end_dt

def _fallback_trial_time(trial_dir: Path) -> Optional[float]:
    """Fallback heuristic if run_meta.json lacks timing."""
    try:
        # mtime of dir as start proxy
        start = datetime.fromtimestamp(trial_dir.stat().st_mtime, tz=timezone.utc)


        # choose the newest artifact present
        candidates = [
            trial_dir / "metrics.json",
            trial_dir / "model.pt",
            trial_dir / "metrics" / "metrics.json",
            trial_dir / "metrics.csv",  # NEW: written even for pruned trials
        ]

        mtimes = []
        for p in candidates:
            if p.exists():
                mtimes.append(datetime.fromtimestamp(p.stat().st_mtime, tz=timezone.utc))
        if not mtimes:
            # nothing better → assume 0 duration
            return None
        end = max(mtimes)
        dur = (end - start).total_seconds()
        return max(dur, 0.0)
    except Exception:
        return None

def _format_hhmmss(seconds: float) -> str:
    td = timedelta(seconds=int(round(seconds)))
    # Force HH:MM:SS even for >24h
    total_seconds = int(td.total_seconds())
    hh = total_seconds // 3600
    mm = (total_seconds % 3600) // 60
    ss = total_seconds % 60
    return f"{hh:02d}:{mm:02d}:{ss:02d}"

# ADD just below _format_hhmmss
def _ceil_to_nearest(seconds: float, base: int = 60) -> int:
    """Ceil seconds to the nearest multiple of `base` (default: 60s = 1 minute)."""
    return int(base * math.ceil(float(seconds) / base))


# -------------------- main --------------------

def main():
    ap = argparse.ArgumentParser(description="Analyze an Optuna HPO job array by JOBID or job root.")
    ap.add_argument("--job-id", help="SLURM JOBID of the array (preferred).")
    ap.add_argument("--job-root", help="Path to outputs/optuna/<job_tag>_NA (or the symlink with JOBID).")
    args = ap.parse_args()

    job_root: Optional[Path] = None
    if args.job_root:
        job_root = Path(args.job_root).resolve()
    elif args.job_id:
        job_root = _resolve_job_root_by_jobid(args.job_id)

    if not job_root or not job_root.exists():
        raise SystemExit("Could not resolve job root. Provide --job-root or a valid --job-id.")

    trials_dir = job_root / "trials"
    if not trials_dir.exists():
        raise SystemExit(f"No trials/ directory under: {job_root}")

    # Collect durations and global start/end
    trial_secs: List[float] = []
    trial_starts: List[datetime] = []
    trial_ends: List[datetime] = []
    missing = 0

    for td in sorted(p for p in trials_dir.iterdir() if p.is_dir()):
        run_meta_path = td / "run_meta.json"
        t_sec = None
        start_dt = None
        end_dt = None

        if run_meta_path.exists():
            try:
                rm = json.loads(run_meta_path.read_text())
                t_sec, start_dt, end_dt = _trial_time_from_run_meta(rm)
            except Exception:
                pass

        if t_sec is None and (start_dt is None or end_dt is None):
            # Use fallback if needed
            t_sec = _fallback_trial_time(td)

        if t_sec is not None:
            trial_secs.append(float(t_sec))
        else:
            missing += 1

        if start_dt:
            trial_starts.append(start_dt)
        if end_dt:
            trial_ends.append(end_dt)

    if not trial_secs and not (trial_starts and trial_ends):
        raise SystemExit("No usable timing information found in trial folders.")

    # Array duration from global earliest start to latest end (if available)
    array_elapsed_sec = None
    if trial_starts and trial_ends:
        array_elapsed_sec = max(trial_ends).timestamp() - min(trial_starts).timestamp()

    # Stats over per-trial durations
    if trial_secs:
        t_min = min(trial_secs)
        t_max = max(trial_secs)
        t_mean = stats.mean(trial_secs)
        t_std = stats.pstdev(trial_secs) if len(trial_secs) > 1 else 0.0

        # Empirical percentiles (no numpy): use statistics.quantiles if available
        # P90 ~ index 89, P95 ~ index 94 with n=100
        p90_emp = p95_emp = None
        try:
            q = stats.quantiles(trial_secs, n=100, method="inclusive")
            # q[k] corresponds to (k+1)th percentile cut; 90th is index 89, 95th is 94
            p90_emp = q[89]
            p95_emp = q[94]
        except Exception:
            # Fallback: simple sort + index approximation
            s = sorted(trial_secs)
            def _pct(s, p):
                if not s:
                    return None
                idx = max(0, min(len(s)-1, int(math.ceil(p * len(s)) - 1)))
                return s[idx]
            p90_emp = _pct(s, 0.90)
            p95_emp = _pct(s, 0.95)

        # Normal approximation to 95th percentile
        z95 = 1.645
        p95_norm = t_mean + z95 * t_std

        # Suggest wall time:
        #   - cover empirical P95 with +10% buffer
        #   - also consider normal approx with +5% buffer
        #   - never below max observed
        #   - add 60s startup, then round up to next minute
        candidates = []
        if p95_emp is not None:
            candidates.append(1.10 * p95_emp)
        candidates.append(1.05 * p95_norm)
        candidates.append(t_max)

        suggested_raw = max(candidates) + 60.0  # +60s startup
        t_suggested_sec = _ceil_to_nearest(suggested_raw, base=60)
    else:
        t_min = t_max = t_mean = t_std = 0.0
        p90_emp = p95_emp = None
        p95_norm = 0.0
        t_suggested_sec = 0

    # Print report
    print(f"Job root         : {job_root}")
    if args.job_id:
        print(f"JOBID            : {args.job_id}")
    print(f"Trials (counted) : {len(trial_secs)}  | missing time info: {missing}")

    if array_elapsed_sec is not None and array_elapsed_sec >= 0:
        print(f"Array elapsed    : {_format_hhmmss(array_elapsed_sec)}  ({int(array_elapsed_sec)} sec)")
    else:
        print("Array elapsed    : n/a")

    print("Trial time (sec) :")
    print(f"  min   = {t_min:.2f}")
    print(f"  max   = {t_max:.2f}")
    print(f"  mean  = {t_mean:.2f}")
    print(f"  std   = {t_std:.2f}")
    if trial_secs:
        if p90_emp is not None:
            print(f"  p90   = {p90_emp:.2f}")
        if p95_emp is not None:
            print(f"  p95   = {p95_emp:.2f} (empirical)")
        print(f"  p95~N = {p95_norm:.2f} (normal approx)")
        print(f"Suggested -t     : {_format_hhmmss(t_suggested_sec)}   # sbatch -t {_format_hhmmss(t_suggested_sec)}")


if __name__ == "__main__":
    main()
