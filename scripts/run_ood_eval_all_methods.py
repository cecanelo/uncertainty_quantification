#!/usr/bin/env python
"""
Run OOD evaluations for all UQ methods (base/ensembles, NF, DIDO) driven
by a single YAML config.

Config: configs/ood_eval_all_methods.yaml

Core ideas
----------
* For each method family we auto-discover run directories that:
    - live under the specified roots,
    - contain a used_config.yaml,
    - have `run_tag` in the run directory name (e.g. "_m_", "_xs_").
* For each discovered run and each enabled OOD dataset, we call the
  appropriate eval script:
    - regression: eval_regression_ood.py
    - nf       : eval_flows_ood.py
    - dido     : eval_dido_ood.py
* Each eval script uses the exact `used_config.yaml`, `preproc_meta.json`
  and `model.pt` from that run (and its base run for NF/DIDO), so ID and
  OOD are evaluated under identical model + preprocessing settings.

Example usage
-------------
Dry run (see what will be executed):

  python scripts/run_ood_eval_all_methods.py \\
      --config configs/ood_eval_all_methods.yaml \\
      --run-tag "_m_" \\
      --dry-run

Run everything for tag "_m_":

  python scripts/run_ood_eval_all_methods.py \\
      --config configs/ood_eval_all_methods.yaml \\
      --run-tag "_m_"

Only NF + DIDO on geo + tail_multi:

  python scripts/run_ood_eval_all_methods.py \\
      --config configs/ood_eval_all_methods.yaml \\
      --run-tag "_m_" \\
      --only-methods nf dido \\
      --only-ood geo tail_multi
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import yaml


def _load_yaml(path: Path) -> Dict[str, Any]:
    return yaml.safe_load(path.read_text()) or {}


def _method_enabled(cfg: Dict[str, Any], name: str) -> bool:
    m = (cfg.get("methods", {}) or {}).get(name, {}) or {}
    return bool(m.get("enabled", False))


def _method_roots(cfg: Dict[str, Any], name: str) -> List[Path]:
    m = (cfg.get("methods", {}) or {}).get(name, {}) or {}
    roots = m.get("roots", ["outputs/trainings"])
    return [Path(r).resolve() for r in roots]


def _regression_head_types(cfg: Dict[str, Any]) -> List[str]:
    m = (cfg.get("methods", {}) or {}).get("regression", {}) or {}
    return [str(h).lower() for h in m.get("head_types", ["point", "laplace", "gauss"])]


def _discover_used_configs(roots: Iterable[Path], run_tag: str) -> List[Tuple[str, Path, Path]]:
    """
    Scan roots for run directories containing a used_config.yaml whose run
    directory name includes run_tag. Returns a list of
        (run_name, run_dir, used_config_path).
    """
    results: List[Tuple[str, Path, Path]] = []
    for root in roots:
        if not root.exists():
            continue
        for cfg_path in root.rglob("used_config.yaml"):
            run_dir = cfg_path.parent
            name = run_dir.name
            if run_tag and run_tag not in name:
                continue
            results.append((name, run_dir, cfg_path))
    return results


def _filter_regression_runs(
    candidates: List[Tuple[str, Path, Path]],
    head_types: List[str],
) -> List[Tuple[str, Path, Path]]:
    """
    Keep only base/ensemble regression runs:
      - parent directory looks like training_point_*, training_lpl_*,
        training_gau_*, or training_ensemble_*,
      - the config's model.head_type is in head_types.
    """
    import yaml as _yaml

    allowed_prefixes = (
        "training_point_",
        "training_lpl_",
        "training_gau_",
        "training_ensemble_",
    )
    out: List[Tuple[str, Path, Path]] = []
    for name, run_dir, cfg_path in candidates:
        parents = {p.name for p in run_dir.parents}
        if not any(p.startswith(allowed_prefixes) for p in parents):
            continue
        try:
            cfg = _yaml.safe_load(cfg_path.read_text()) or {}
            ht = str((cfg.get("model", {}) or {}).get("head_type", "")).lower()
        except Exception:
            ht = ""
        if ht not in head_types:
            continue
        out.append((name, run_dir, cfg_path))
    return out


def _filter_nf_runs(candidates: List[Tuple[str, Path, Path]]) -> List[Tuple[str, Path, Path]]:
    """
    Keep only NF runs: those whose parent directory name starts with
    'training_nf_'.
    """
    out: List[Tuple[str, Path, Path]] = []
    for name, run_dir, cfg_path in candidates:
        parents = {p.name for p in run_dir.parents}
        if any(p.startswith("training_nf_") for p in parents):
            out.append((name, run_dir, cfg_path))
    return out


def _filter_dido_runs(candidates: List[Tuple[str, Path, Path]]) -> List[Tuple[str, Path, Path]]:
    """
    Keep only DIDO runs: those whose parent directory name starts with
    'training_dido_'.
    """
    out: List[Tuple[str, Path, Path]] = []
    for name, run_dir, cfg_path in candidates:
        parents = {p.name for p in run_dir.parents}
        if any(p.startswith("training_dido_") for p in parents):
            out.append((name, run_dir, cfg_path))
    return out


def _submit_slurm(
    cfg_path: Path,
    cfg: Dict[str, Any],
    args: argparse.Namespace,
    run_tag: str,
) -> None:
    """
    Submit a single SLURM job that will run this script in local mode
    inside the cluster environment. SLURM options are read from
    config['slurm'] with sensible defaults.
    """
    slurm = (cfg.get("slurm", {}) or {})
    partition = slurm.get("partition", "TEST")
    time_str = slurm.get("time", "02:00:00")
    mem_gb = int(slurm.get("mem_gb", 32))
    cpus = int(slurm.get("cpus", 4))
    gpus = int(slurm.get("gpus", 0))
    job_name = slurm.get("job_name", "ood_all")
    conda_env = slurm.get("conda_env", "thesis")

    repo_root = Path(__file__).resolve().parents[1]
    logs_root = (repo_root / "logs" / "ood").resolve()
    logs_root.mkdir(parents=True, exist_ok=True)

    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    out_log = logs_root / f"{job_name}_{ts}_%j.out"
    err_log = logs_root / f"{job_name}_{ts}_%j.err"
    gres_line = f"#SBATCH --gres=gpu:{gpus}" if gpus > 0 else ""

    # Command to run inside the job: this same script in local mode
    cmd_parts = [
        f"python \"{(repo_root / 'scripts' / 'run_ood_eval_all_methods.py')}\"",
        f"--config \"{cfg_path}\"",
        f"--run-tag \"{run_tag}\"",
    ]
    if args.only_methods:
        methods_str = " ".join(args.only_methods)
        cmd_parts.append(f"--only-methods {methods_str}")
    if args.only_ood:
        ood_str = " ".join(args.only_ood)
        cmd_parts.append(f"--only-ood {ood_str}")
    # We intentionally do not pass --mode or --dry-run; inside the job
    # the script will run in local mode and execute the eval commands.
    cmd_line = " ".join(cmd_parts)

    script = f"""#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --partition={partition}
#SBATCH --cpus-per-task={cpus}
#SBATCH --mem={mem_gb}G
{gres_line}
#SBATCH --time={time_str}
#SBATCH --output={out_log}
#SBATCH --error={err_log}

cd "{repo_root}"
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate {conda_env}
echo "[env] host=$(hostname) date=$(date)"
{cmd_line}
"""
    print("[ood_all][slurm] sbatch script:\n")
    print(script)
    if args.dry_run:
        return

    res = subprocess.run(["sbatch"], input=script.encode("utf-8"), check=False, capture_output=True)
    if res.returncode != 0:
        print(res.stdout.decode())
        print(res.stderr.decode(), file=sys.stderr)
        res.check_returncode()
    else:
        print(res.stdout.decode().strip())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--config",
        type=str,
        default="configs/ood_eval_all_methods.yaml",
        help="YAML config describing methods and OOD sets.",
    )
    ap.add_argument(
        "--run-tag",
        type=str,
        default=None,
        help="Substring used to select runs (overrides config.run_tag if set).",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands instead of executing them.",
    )
    ap.add_argument(
        "--only-methods",
        nargs="*",
        choices=["regression", "nf", "dido"],
        default=None,
        help="Optional subset of methods to run.",
    )
    ap.add_argument(
        "--only-ood",
        nargs="*",
        default=None,
        help="Optional subset of OOD dataset keys to run.",
    )
    ap.add_argument(
        "--mode",
        choices=["local", "slurm"],
        default="local",
        help="Run locally (default) or submit a single SLURM job.",
    )
    args = ap.parse_args()

    cfg_path = Path(args.config).resolve()
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config not found: {cfg_path}")
    cfg = _load_yaml(cfg_path)

    run_tag = args.run_tag or cfg.get("run_tag", "_m_")
    if args.mode == "slurm":
        _submit_slurm(cfg_path, cfg, args, run_tag)
        return

    ood_sets = cfg.get("ood_sets", {}) or {}
    if not ood_sets:
        raise ValueError("No ood_sets defined in config.")

    # Prepare OOD specs
    ood_specs: List[Tuple[str, Path]] = []
    for key, ospec in ood_sets.items():
        if not ospec.get("enabled", True):
            continue
        if args.only_ood and key not in args.only_ood:
            continue
        csv_path = Path(ospec["csv"]).resolve()
        if not csv_path.exists():
            print(f"[warn] OOD CSV for {key} does not exist: {csv_path}")
            continue
        ood_specs.append((key, csv_path))
    if not ood_specs:
        print("[info] No OOD sets enabled or found; nothing to do.")
        return

    # ---------------- Regression (base + ensembles) ----------------
    if _method_enabled(cfg, "regression") and (not args.only_methods or "regression" in args.only_methods):
        reg_roots = _method_roots(cfg, "regression")
        candidates = _discover_used_configs(reg_roots, run_tag)
        reg_head_types = _regression_head_types(cfg)
        reg_runs = _filter_regression_runs(candidates, reg_head_types)
        print(f"[discover][regression] {len(reg_runs)} runs for tag '{run_tag}'")

        for run_name, run_dir, cfg_p in reg_runs:
            for ood_key, ood_csv in ood_specs:
                cmd = [
                    sys.executable,
                    "scripts/eval_regression_ood.py",
                    "--config",
                    str(cfg_p),
                    "--outdir",
                    str(run_dir),
                    "--ood-csv",
                    str(ood_csv),
                    "--dataset-label",
                    ood_key,
                ]
                print(" ".join(cmd))
                if not args.dry_run:
                    subprocess.run(cmd, check=True)

    # ---------------- NF (normalizing flows) ----------------
    if _method_enabled(cfg, "nf") and (not args.only_methods or "nf" in args.only_methods):
        nf_roots = _method_roots(cfg, "nf")
        candidates = _discover_used_configs(nf_roots, run_tag)
        nf_runs = _filter_nf_runs(candidates)
        print(f"[discover][nf] {len(nf_runs)} runs for tag '{run_tag}'")

        import sys as _sys

        for run_name, run_dir, cfg_p in nf_runs:
            for ood_key, ood_csv in ood_specs:
                cmd = [
                    _sys.executable,
                    "scripts/eval_flows_ood.py",
                    "--config",
                    str(cfg_p),
                    "--outdir",
                    str(run_dir),
                    "--ood-csv",
                    str(ood_csv),
                    "--dataset-label",
                    ood_key,
                ]
                print(" ".join(cmd))
                if not args.dry_run:
                    subprocess.run(cmd, check=True)

    # ---------------- DIDO ----------------
    if _method_enabled(cfg, "dido") and (not args.only_methods or "dido" in args.only_methods):
        dido_roots = _method_roots(cfg, "dido")
        candidates = _discover_used_configs(dido_roots, run_tag)
        dido_runs = _filter_dido_runs(candidates)
        print(f"[discover][dido] {len(dido_runs)} runs for tag '{run_tag}'")

        import sys as _sys

        for run_name, run_dir, _cfg_p in dido_runs:
            for ood_key, ood_csv in ood_specs:
                cmd = [
                    _sys.executable,
                    "scripts/eval_dido_ood.py",
                    "--dido-outdir",
                    str(run_dir),
                    "--ood-csv",
                    str(ood_csv),
                    "--dataset-label",
                    ood_key,
                ]
                print(" ".join(cmd))
                if not args.dry_run:
                    subprocess.run(cmd, check=True)

    print("Done.")


if __name__ == "__main__":
    import sys

    main()
