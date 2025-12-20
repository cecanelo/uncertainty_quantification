#!/usr/bin/env python3
"""
Launch a kNN OOD sweep with unique outputs per trial.

What it does
------------
- Reads a sweep config (YAML) like configs/hpo_knn_ood.yaml.
- Forces candidate_split=all and ref_limit=None unless overridden.
- Writes a unique per-trial config under logs/knn_ood_configs/.
- Sets a unique out_prefix per trial (craigslist_<name>) to avoid overwriting outputs.
- Runs make_knn_ood_knn.py for each trial (locally or via slurm, per run.mode).

Usage
-----
python scripts/run_knn_ood_sweep.py --config configs/hpo_knn_ood.yaml
python scripts/run_knn_ood_sweep.py --config configs/hpo_knn_ood.yaml --run-mode slurm
python scripts/run_knn_ood_sweep.py --config configs/hpo_knn_ood.yaml --dry-run
"""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path
from typing import Any, Dict, List

import yaml


def load_config(path: Path) -> Dict[str, Any]:
    cfg = yaml.safe_load(path.read_text())
    if not isinstance(cfg, dict):
        raise ValueError("Config must be a mapping.")
    return cfg


def main() -> None:
    ap = argparse.ArgumentParser(description="Run kNN OOD sweep with unique outputs per trial.")
    ap.add_argument("--config", required=True, type=Path, help="Path to sweep YAML (e.g., configs/hpo_knn_ood.yaml)")
    ap.add_argument("--run-mode", choices=["local", "slurm"], default=None, help="Override run.mode for all trials")
    ap.add_argument("--out-config-dir", type=Path, default=Path("logs/knn_ood_configs"), help="Where to write per-trial configs")
    ap.add_argument("--dry-run", action="store_true", help="Print commands without running")
    args = ap.parse_args()

    cfg = load_config(args.config)
    base_run: Dict[str, Any] = cfg.get("run", {})
    base_knn: Dict[str, Any] = cfg.get("knn_ood", {})
    sweep: List[Dict[str, Any]] = cfg.get("sweep", [])

    # Force full dataset candidates unless explicitly overridden
    base_knn.setdefault("candidate_split", "all")
    base_knn.setdefault("ref_limit", None)

    out_cfg_dir = args.out_config_dir.expanduser().resolve()
    out_cfg_dir.mkdir(parents=True, exist_ok=True)

    for i, trial in enumerate(sweep):
        name = trial.get("name") or f"trial_{i}"
        run = base_run.copy()
        knn = base_knn.copy()
        knn.update(trial.get("knn_ood", {}))

        # Unique outputs per trial
        knn["out_prefix"] = knn.get("out_prefix", f"craigslist_{name}")
        run["job_name"] = f"knn_ood_{name}"

        trial_cfg = {"run": run, "knn_ood": knn}
        cfg_path = out_cfg_dir / f"knn_ood_{name}.yaml"
        cfg_path.write_text(yaml.safe_dump(trial_cfg))

        mode = args.run_mode or run.get("mode", "local")
        cmd = ["python", "scripts/make_knn_ood_knn.py", "--config", str(cfg_path), "--run-mode", mode]

        print(f"\n=== {name} ===")
        print(" ".join(cmd))
        if not args.dry_run:
            subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
