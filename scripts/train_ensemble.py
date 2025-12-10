#!/usr/bin/env python3
"""
Train an ensemble of regression models by launching `train_regression.py` once per member.

Supports:
  - local mode: sequentially trains all members in-process.
  - slurm mode: submits an array job, one task per member (like HPO runners).

Config expectations:
  - Base train config (YAML) should include an `ensemble` block:
      ensemble:
        enabled: false
        n_members: 5
  - When run in ensemble mode, we read `n_members` from that block.
  - Each member gets its own seed: base seed + member_idx (applied to cfg["seed"] and data.split_seed).
  - Member outdirs are under the specified ensemble_root: ensemble_root/member_{idx:02d}

Usage examples:
  Local:
    python scripts/train_ensemble.py --config configs/train_laplace.yaml \
        --ensemble-root outputs/ensembles/laplace_test --mode local

  Slurm:
    python scripts/train_ensemble.py --config configs/train_laplace.yaml \
        --ensemble-root outputs/ensembles/laplace_test --mode slurm

Notes:
  - We intentionally leave `train_regression.py` unchanged; this script just orchestrates.
  - The base config is snapshotted into ensemble_root/base_config.yaml for reproducibility.
  - If --ensemble-root is omitted, defaults to outputs/trainings/ensembles/<job|cfg>_<timestamp>.
"""
from __future__ import annotations
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
TRAIN_SCRIPT = REPO_ROOT / "scripts" / "train_regression.py"


def load_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r") as f:
        return yaml.safe_load(f)


def _deep_merge(base: Dict[str, Any], patch: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(base)
    for k, v in (patch or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def _infer_head_type(base_cfg: Dict[str, Any]) -> str:
    try:
        return str(base_cfg.get("model", {}).get("head_type", "")).lower() or "unknown"
    except Exception:
        return "unknown"


def _prefix_job(job: str, head: str) -> str:
    pref = f"esm_{head}_{job}"
    return pref if not job.startswith("esm_") else job


def save_yaml(obj: Dict[str, Any], path: Path) -> None:
    with path.open("w") as f:
        yaml.safe_dump(obj, f, sort_keys=False)


def _member_outdir(root: Path, idx: int) -> Path:
    return root / f"member_{idx:02d}"


def _build_member_config(
    base_cfg: Dict[str, Any],
    member_idx: int,
    seed: int,
    member_outdir: Path,
    overrides: Dict[str, Any] | None = None,
    evals_root: Path | None = None,
) -> Dict[str, Any]:
    cfg = json.loads(json.dumps(base_cfg))  # deep copy via JSON for simplicity
    if overrides:
        cfg = _deep_merge(cfg, overrides)

    cfg["seed"] = int(seed)
    cfg.setdefault("data", {})
    cfg["data"]["split_seed"] = int(seed)

    # Wire IO outdir for preproc/meta and model artifacts
    cfg.setdefault("io", {})
    cfg["io"]["outdir"] = str(member_outdir)
    if evals_root is not None:
        cfg["io"]["evals_root"] = str(evals_root)

    # Leave eval_after_train as-is (user-controlled)
    return cfg


def _run_member_local(cfg_path: Path, outdir: Path) -> None:
    cmd = [sys.executable, str(TRAIN_SCRIPT), "--config", str(cfg_path), "--outdir", str(outdir)]
    print(f"[ensemble][local] Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def _write_manifest(root: Path, base_cfg_path: Path, n_members: int, mode: str, ts: str) -> None:
    manifest = {
        "base_config": str(base_cfg_path),
        "n_members": int(n_members),
        "mode": mode,
        "timestamp": ts,
    }
    with (root / "ensemble_manifest.json").open("w") as f:
        json.dump(manifest, f, indent=4)


def _resolve_root(raw: str, cfg_path: Path, base_cfg: Dict[str, Any]) -> Path:
    ts_env = os.environ.get("ENSEMBLE_TS")
    ts = ts_env if ts_env else datetime.now().strftime("%Y%m%d-%H%M%S")
    jobid_env = os.environ.get("SLURM_ARRAY_JOB_ID") or os.environ.get("SLURM_JOB_ID") or "NA"
    head = _infer_head_type(base_cfg)
    job_raw = base_cfg.get("slurm", {}).get("job_name") or cfg_path.stem
    job = _prefix_job(job_raw, head)
    filled = (
        raw.replace("{job}", str(job))
           .replace("{ts}", ts)
           .replace("{jobid}", jobid_env)
    )
    return Path(filled).resolve()


def submit_slurm_array(base_cfg_path: Path, ensemble_root_tpl: str, n_members: int, base_cfg: Dict[str, Any]) -> None:
    head = _infer_head_type(base_cfg)
    # Pull slurm defaults from base config
    slurm_cfg = base_cfg.get("slurm", {}) or {}
    partition = slurm_cfg.get("partition", "TEST")
    time_str = slurm_cfg.get("time", "01:00:00")
    mem_gb = int(slurm_cfg.get("mem_gb", 16))
    cpus = int(slurm_cfg.get("cpus", 2))
    gpus = int(slurm_cfg.get("gpus", 0))
    conda_env = slurm_cfg.get("conda_env", "thesis")
    # Derive job/tag from the base config path: prefer parent folder (actual run tag), else filename stem.
    base_parent = base_cfg_path.parent.name
    job_base = base_parent if base_parent and base_parent != "." else base_cfg_path.stem
    job_name = _prefix_job(job_base, head)
    logs_root = Path("logs/train").resolve()
    logs_root.mkdir(parents=True, exist_ok=True)

    ts = datetime.now().strftime("%Y%m%d-%H%M%S")

    # Log stub matches ensemble tag: <job>_<ts>_<JOBID>_<task>
    log_stub = f"{job_name}_{ts}_%A"
    out_log = logs_root / f"{log_stub}_%a.out"
    err_log = logs_root / f"{log_stub}_%a.err"

    array_spec = f"0-{n_members - 1}"
    max_parallel = int(slurm_cfg.get("max_parallel", n_members))
    if n_members > 1 and 1 <= max_parallel < n_members:
        array_spec = f"{array_spec}%{max_parallel}"

    script_path = Path(__file__).resolve()
    this_script = script_path

    lines = [
        "#!/bin/bash",
        f"#SBATCH --job-name={job_name}",
        f"#SBATCH --partition={partition}",
        f"#SBATCH --cpus-per-task={cpus}",
        f"#SBATCH --mem={mem_gb}G",
        f"#SBATCH --time={time_str}",
        f"#SBATCH --output={out_log}",
        f"#SBATCH --error={err_log}",
        f"#SBATCH --array={array_spec}",
    ]
    if gpus > 0:
        lines.insert(6, f"#SBATCH --gres=gpu:{gpus}")

    exports = [
        f"ENSEMBLE_CONFIG={base_cfg_path}",
        f"ENSEMBLE_ROOT_TEMPLATE={ensemble_root_tpl}",
        f"N_MEMBERS={n_members}",
        f"ENSEMBLE_TS={ts}",
    ]
    lines += [
        f"export {' '.join(exports)}",
        f"cd '{REPO_ROOT}'",
        'source "$HOME/miniconda3/etc/profile.d/conda.sh"',
        f"conda activate {conda_env}",
        'echo "[env] host=$(hostname) date=$(date) array=$SLURM_ARRAY_TASK_ID"',
        f"python '{this_script}' --config '{base_cfg_path}' --mode array-worker",
    ]

    sb_script = "\n".join(lines) + "\n"
    print("[ensemble][slurm] sbatch script:\n")
    print(sb_script)

    res = subprocess.run(["sbatch"], input=sb_script.encode("utf-8"), check=False, capture_output=True)
    if res.returncode != 0:
        print(res.stdout.decode())
        print(res.stderr.decode(), file=sys.stderr)
        res.check_returncode()
    else:
        print(res.stdout.decode().strip())


def main() -> None:
    ap = argparse.ArgumentParser(description="Train an ensemble of models using train_regression.py")
    ap.add_argument("--config", help="Base train config YAML (required unless --wrapper is provided)")
    ap.add_argument("--wrapper", help="Optional wrapper YAML with base_config, ensemble settings, io.ensemble_root")
    ap.add_argument("--ensemble-root", required=False, help="Root folder for ensemble (default from wrapper or derived template)")
    ap.add_argument("--mode", choices=["local", "slurm", "array-worker"], default=None,
                    help="local: run all members here; slurm: submit array; array-worker: internal (wrapper.run.mode can set default)")
    ap.add_argument("--base-seed", type=int, default=None, help="Override base seed (else uses cfg['seed'])")
    args = ap.parse_args()

    wrapper_cfg = load_yaml(Path(args.wrapper)) if args.wrapper else None
    if wrapper_cfg:
        if "base_config" not in wrapper_cfg:
            raise SystemExit("Wrapper config must specify base_config")
        cfg_path = Path(wrapper_cfg["base_config"]).resolve()
    elif args.config:
        cfg_path = Path(args.config).resolve()
    else:
        raise SystemExit("Provide --config or --wrapper")

    base_cfg = load_yaml(cfg_path)

    # Allow wrapper to override SLURM settings (e.g., conda_env) so users don't have to edit base config
    if wrapper_cfg and "slurm" in wrapper_cfg:
        merged_slurm = dict(base_cfg.get("slurm", {}) or {})
        # Avoid overriding the base job_name so tags/logs follow the base config
        wrapper_slurm = dict(wrapper_cfg.get("slurm", {}) or {})
        wrapper_slurm.pop("job_name", None)
        merged_slurm.update(wrapper_slurm)
        base_cfg["slurm"] = merged_slurm

    run_cfg = wrapper_cfg.get("run", {}) if wrapper_cfg else {}
    run_mode = args.mode or run_cfg.get("mode", "local")

    # Ensemble root resolution (CLI > wrapper > default).
    if run_mode == "slurm":
        if args.ensemble_root:
            ensemble_root = args.ensemble_root
        elif wrapper_cfg and wrapper_cfg.get("io", {}).get("ensemble_root"):
            ensemble_root = wrapper_cfg["io"]["ensemble_root"]
        else:
            head = _infer_head_type(base_cfg)
            tag_raw = base_cfg.get("slurm", {}).get("job_name") or cfg_path.stem
            tag = _prefix_job(tag_raw, head)
            ensemble_root = str(REPO_ROOT / "outputs" / "trainings" / "ensembles" / f"{tag}_{'{ts}'}_{'{jobid}'}")
        # do not mkdir here; worker will resolve jobid and mkdir
    elif run_mode == "array-worker":
        tmpl = os.environ.get("ENSEMBLE_ROOT_TEMPLATE")
        if not tmpl:
            raise SystemExit("array-worker mode requires ENSEMBLE_ROOT_TEMPLATE")
        ensemble_root = _resolve_root(tmpl, cfg_path, base_cfg)
        Path(ensemble_root).mkdir(parents=True, exist_ok=True)
    else:
        if args.ensemble_root:
            ensemble_root = _resolve_root(args.ensemble_root, cfg_path, base_cfg)
        elif wrapper_cfg and wrapper_cfg.get("io", {}).get("ensemble_root"):
            ensemble_root = _resolve_root(wrapper_cfg["io"]["ensemble_root"], cfg_path, base_cfg)
        else:
            head = _infer_head_type(base_cfg)
            tag_raw = base_cfg.get("slurm", {}).get("job_name") or cfg_path.stem
            tag = _prefix_job(tag_raw, head)
            ts_root = datetime.now().strftime("%Y%m%d-%H%M%S")
            ensemble_root = (REPO_ROOT / "outputs" / "trainings" / "ensembles" / f"{tag}_{ts_root}").resolve()
        Path(ensemble_root).mkdir(parents=True, exist_ok=True)

    # n_members (wrapper override > base config)
    if wrapper_cfg and "ensemble" in wrapper_cfg and "n_members" in wrapper_cfg["ensemble"]:
        n_members = int(wrapper_cfg["ensemble"]["n_members"])
    else:
        ens_cfg = base_cfg.get("ensemble", {}) or {}
        n_members = int(ens_cfg.get("n_members", 1))
    if n_members < 1:
        raise SystemExit("ensemble.n_members must be >= 1")

    # If running as array-worker, prefer env-provided N_MEMBERS/base_seed (exported from submit step)
    if run_mode == "array-worker":
        n_members_env = os.environ.get("N_MEMBERS")
        if n_members_env:
            n_members = int(n_members_env)

    # base seed (CLI > wrapper > base config)
    if args.base_seed is not None:
        base_seed = args.base_seed
    elif wrapper_cfg and "ensemble" in wrapper_cfg and "base_seed" in wrapper_cfg["ensemble"]:
        base_seed = int(wrapper_cfg["ensemble"]["base_seed"])
    else:
        base_seed = int(base_cfg.get("seed", 42))

    # Slurm submission: defer all work to array workers; no snapshot here
    if run_mode == "slurm":
        submit_slurm_array(cfg_path, ensemble_root, n_members, base_cfg)
        return

    ensemble_tag = Path(ensemble_root).name
    evals_root_cfg = (wrapper_cfg.get("io", {}).get("evals_root") if wrapper_cfg else None)
    if evals_root_cfg:
        evals_root = (
            str(evals_root_cfg)
            .replace("{ensemble}", ensemble_tag)
            .replace("{job}", ensemble_tag)
            .replace("{jobid}", os.environ.get("SLURM_JOB_ID", "NA"))
        )
        evals_root = Path(evals_root).resolve()
    else:
        evals_root = (REPO_ROOT / "outputs" / "evals" / ensemble_tag).resolve()

    member_overrides = (wrapper_cfg.get("member_overrides") if wrapper_cfg else None)

    # Snapshot base config (local/array-worker)
    snapshot_path = Path(ensemble_root) / "base_config.yaml"
    if not snapshot_path.exists():
        save_yaml(base_cfg, snapshot_path)

    # Manifest (only write once in non-worker; in worker only if missing)
    if run_mode != "array-worker":
        ts = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
        _write_manifest(Path(ensemble_root), snapshot_path, n_members, run_mode, ts)
    else:
        manifest_path = Path(ensemble_root) / "ensemble_manifest.json"
        if not manifest_path.exists():
            ts = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
            _write_manifest(Path(ensemble_root), snapshot_path, n_members, run_mode, ts)

    if run_mode == "array-worker":
        task_id_str = os.environ.get("SLURM_ARRAY_TASK_ID")
        if task_id_str is None:
            raise SystemExit("array-worker mode requires SLURM_ARRAY_TASK_ID")
        m_idx = int(task_id_str)
        if m_idx < 0 or m_idx >= n_members:
            raise SystemExit(f"Invalid member index {m_idx} for n_members={n_members}")
        seed = base_seed + m_idx
        member_outdir = _member_outdir(ensemble_root, m_idx)
        member_outdir.mkdir(parents=True, exist_ok=True)
        member_cfg = _build_member_config(base_cfg, m_idx, seed, member_outdir, member_overrides, evals_root)
        member_cfg_path = member_outdir / "member_config.yaml"
        save_yaml(member_cfg, member_cfg_path)
        _run_member_local(member_cfg_path, member_outdir)
        return

    # local mode: loop over all members sequentially
    for m_idx in range(n_members):
        seed = base_seed + m_idx
        member_outdir = _member_outdir(ensemble_root, m_idx)
        member_outdir.mkdir(parents=True, exist_ok=True)
        member_cfg = _build_member_config(base_cfg, m_idx, seed, member_outdir, member_overrides, evals_root)
        member_cfg_path = member_outdir / "member_config.yaml"
        save_yaml(member_cfg, member_cfg_path)
        _run_member_local(member_cfg_path, member_outdir)


if __name__ == "__main__":
    main()
