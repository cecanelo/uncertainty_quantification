#!/usr/bin/env python3
"""
Build a kNN-based ID / hard-OOD split from a cleaned Craigslist CSV.

What it does
------------
- Uses stored preprocessing metadata (preproc_meta.json) to project rows into the
  same feature space used for training (numeric + onehot + hashed).
- Applies a random Gaussian projection to reduce dimensionality (default: 32D).
- Uses train split embeddings as the kNN reference set; scores the candidate split
  (default: test) by mean distance to the K nearest reference neighbors.
- Writes two CSVs with knn_score attached:
    * <out_prefix>_id_knn.csv     (rows in low-distance quantile band)
    * <out_prefix>_ood_knn.csv    (rows in high-distance quantile band)

Config (YAML) shape
-------------------
run:
  mode: "slurm"            # "local" or "slurm"
  partition: "STUD"
  time: "00:30:00"
  mem_gb: 16
  cpus: 4
  gpus: 0
  conda_env: "thesis"
  job_name: "knn_ood"
  logs_dir: "logs"

knn_ood:
  csv_path: "datasets/craigslist_cleaned_FULL.csv"
  preproc_meta: "outputs/trainings/training_point_m/.../preproc_meta.json"
  out_dir: "datasets/knn_ood"
  out_prefix: "craigslist"
  ref_split: "train"
  candidate_split: "test"
  ref_limit: 50000          # optional cap on reference rows
  proj_dim: 32
  knn_k: 5                  # optional single k
  knn_k_list: [3, 5, 8, 12] # optional list to auto-pick k by quantile gap
  id_q: [0.0, 0.80]         # quantile band for ID
  ood_q: [0.95, 1.0]        # quantile band for hard OOD
  chunk_rows: 20000
  seed: 17

Usage
-----
python scripts/make_knn_ood_knn.py --config configs/knn_ood.yaml            # local
python scripts/make_knn_ood_knn.py --config configs/knn_ood.yaml --run-mode local
python scripts/make_knn_ood_knn.py --config configs/knn_ood.yaml --run-mode slurm  # submit sbatch
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import yaml
from sklearn.neighbors import NearestNeighbors

# Ensure project root on sys.path so `scripts.*` imports work regardless of CWD.
import sys
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.data import _hash_bucket


# -----------------------
# Helpers
# -----------------------
def _load_config(path: Path) -> Dict:
    path = path.expanduser().resolve()
    with path.open("r") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError("Config must be a YAML mapping.")
    return cfg


def _add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add the same derived numeric cols used in training."""
    cur_year = pd.Timestamp.now().year
    if "year" in df.columns:
        df["year"] = pd.to_numeric(df["year"], errors="coerce")
    if "odometer" in df.columns:
        df["odometer"] = pd.to_numeric(df["odometer"], errors="coerce")
    if "lat" in df.columns:
        df["lat"] = pd.to_numeric(df["lat"], errors="coerce")
    if "long" in df.columns:
        df["long"] = pd.to_numeric(df["long"], errors="coerce")

    if "year" in df.columns and "odometer" in df.columns:
        age = (cur_year - df["year"]).clip(lower=0)
        df["age_years"] = age
        df["log_odometer"] = np.log1p(df["odometer"].fillna(0))
        df["mileage_per_year"] = df["odometer"].fillna(0) / np.where(age > 0, age, 1)
    return df


def _build_projection(feature_dim: int, proj_dim: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    P = rng.standard_normal(size=(feature_dim, proj_dim), dtype=np.float32)
    P /= np.sqrt(proj_dim)
    return P.astype(np.float32)


def _project_chunk(
    df: pd.DataFrame,
    enc: Dict,
    proj: np.ndarray,
    numeric_cols: List[str],
    onehot_levels: Dict[str, List[str]],
    hash_dims: Dict[str, int],
) -> np.ndarray:
    """
    Fast projection without materializing the full sparse feature matrix.
    """
    n = len(df)
    proj_dim = proj.shape[1]
    Z = np.zeros((n, proj_dim), dtype=np.float32)

    # Precompute onehot index maps once
    oh_index: Dict[str, Dict[str, int]] = {c: {lv: i for i, lv in enumerate(levels)} for c, levels in onehot_levels.items()}

    offset = 0
    # Numeric (standardized)
    for c in numeric_cols:
        slice_p = proj[offset : offset + 1]  # shape (1, proj_dim)
        offset += 1
        if c in df.columns:
            x = pd.to_numeric(df[c], errors="coerce").to_numpy()
            mean = float(enc["num"][c]["mean"])
            std = float(enc["num"][c]["std"]) if enc["num"][c]["std"] > 1e-12 else 1.0
            x = (x - mean) / std
        else:
            x = np.zeros(n, dtype=np.float32)
        Z += x.reshape(-1, 1).astype(np.float32) * slice_p

    # One-hot
    for c, levels in onehot_levels.items():
        idx_map = oh_index[c]
        slice_p = proj[offset : offset + len(levels)]
        offset += len(levels)
        if c not in df.columns:
            continue
        s = df[c].astype(str).fillna("__NA__").to_numpy()
        for i, val in enumerate(s):
            j = idx_map.get(val)
            if j is not None:
                Z[i] += slice_p[j]

    # Hashed (signed hashing)
    for c, dim in hash_dims.items():
        dim = int(dim)
        slice_p = proj[offset : offset + dim]
        offset += dim
        if c not in df.columns:
            continue
        s = df[c].astype(str).fillna("__NA__").to_numpy()
        for i, val in enumerate(s):
            key = f"{c}={val}"
            b = _hash_bucket(key, dim)
            sign = -1.0 if (_hash_bucket(key + "#sign", 2) % 2) else 1.0
            Z[i] += sign * slice_p[b]

    if offset != proj.shape[0]:
        raise RuntimeError(f"Projection dimension mismatch (used {offset}, expected {proj.shape[0]}).")
    return Z


def _maybe_submit_slurm(cfg: Dict, args: argparse.Namespace) -> bool:
    run = cfg.get("run", {})
    if str(run.get("mode", "local")).lower() != "slurm":
        return False

    logs_dir = Path(run.get("logs_dir", "logs")).expanduser()
    logs_dir.mkdir(parents=True, exist_ok=True)

    job_name = run.get("job_name", "knn_ood")
    partition = run.get("partition", "short")
    time = run.get("time", "00:30:00")
    mem_gb = run.get("mem_gb", 16)
    cpus = run.get("cpus", 4)
    gpus = int(run.get("gpus", 0))
    conda_env = run.get("conda_env")

    script_path = Path(__file__).resolve()
    config_path = Path(args.config).resolve()

    sb_lines = [
        "#!/bin/bash",
        f"#SBATCH -J {job_name}",
        f"#SBATCH -p {partition}",
        f"#SBATCH -t {time}",
        f"#SBATCH -c {cpus}",
        f"#SBATCH --mem={mem_gb}G",
        f"#SBATCH -o {logs_dir / (job_name + '-%j.out')}",
        f"#SBATCH -e {logs_dir / (job_name + '-%j.err')}",
    ]
    if gpus > 0:
        sb_lines.append(f"#SBATCH --gpus={gpus}")

    sb_lines.append("")  # blank line
    sb_lines.append("set -euo pipefail")
    if conda_env:
        sb_lines.extend(
            [
                # Try both conda init paths; tolerate clusters without bashrc init
                "source ~/.bashrc 2>/dev/null || true",
                'eval "$(conda shell.bash hook)"',
                f"conda activate {conda_env}",
            ]
        )
    sb_lines.append(f"python {script_path} --config {config_path} --run-mode local")

    sb_script = "\n".join(sb_lines) + "\n"
    print("[knn-ood][slurm] sbatch script:\n")
    print(sb_script)

    if args.dry_run:
        print("[knn-ood][slurm] dry-run: not submitting.")
        return True

    res = subprocess.run(["sbatch"], input=sb_script.encode("utf-8"), check=False, capture_output=True)
    if res.returncode != 0:
        print("[knn-ood][slurm] sbatch submission failed:")
        print(res.stdout.decode("utf-8"))
        print(res.stderr.decode("utf-8"))
    else:
        print(res.stdout.decode("utf-8").strip())
    return True


# -----------------------
# Main builder
# -----------------------
def build_knn_splits(cfg: Dict) -> None:
    params = cfg.get("knn_ood", {})
    csv_path = Path(params["csv_path"]).expanduser().resolve()
    preproc_meta = Path(params["preproc_meta"]).expanduser().resolve()
    out_dir = Path(params.get("out_dir", csv_path.parent)).expanduser().resolve()
    out_prefix = params.get("out_prefix", "craigslist")

    ref_split = params.get("ref_split", "train")
    cand_split = params.get("candidate_split", "test")  # set to "all" to score the full dataset
    ref_limit = params.get("ref_limit")

    proj_dim = int(params.get("proj_dim", 32))
    knn_k_single = int(params.get("knn_k", 5))
    knn_k_list = params.get("knn_k_list")
    if knn_k_list is not None:
        knn_k_list = [int(k) for k in knn_k_list]
    else:
        knn_k_list = [knn_k_single]
    id_q = params.get("id_q", [0.0, 0.8])
    ood_q = params.get("ood_q", [0.95, 1.0])
    chunk_rows = int(params.get("chunk_rows", 20_000))
    seed = int(params.get("seed", 17))

    if not csv_path.exists():
        raise FileNotFoundError(f"csv_path not found: {csv_path}")
    if not preproc_meta.exists():
        raise FileNotFoundError(f"preproc_meta not found: {preproc_meta}")

    meta = json.loads(preproc_meta.read_text())
    id_col = meta.get("id_col", "id")
    id_values = meta.get("id_values")
    if id_values is None:
        raise ValueError("preproc_meta.json is missing 'id_values'; cannot map row indices to ids.")

    # Build id sets for ref and candidate splits
    split_idx = meta.get("splits", {})
    if ref_split not in split_idx:
        raise ValueError(f"ref_split={ref_split} missing in preproc_meta.splits")

    ref_ids = [str(id_values[i]) for i in split_idx[ref_split]]

    if str(cand_split).lower() == "all":
        cand_ids = [str(x) for x in id_values]  # train + val + test
    else:
        if cand_split not in split_idx:
            raise ValueError(f"candidate_split={cand_split} missing in preproc_meta.splits")
        cand_ids = [str(id_values[i]) for i in split_idx[cand_split]]

    if ref_limit is not None and ref_limit < len(ref_ids):
        rng = np.random.default_rng(seed)
        ref_ids = rng.choice(ref_ids, size=int(ref_limit), replace=False).tolist()

    ref_set = set(ref_ids)
    cand_set = set(cand_ids)

    numeric_cols: List[str] = list(meta["encoders"]["num"].keys())
    onehot_levels: Dict[str, List[str]] = meta["encoders"]["oh_levels"]
    hash_dims: Dict[str, int] = meta["encoders"]["hash_dims"]
    feature_dim = int(meta["feature_dim"])
    proj = _build_projection(feature_dim, proj_dim, seed)

    ref_embeddings: List[np.ndarray] = []
    cand_embeddings: List[np.ndarray] = []
    cand_rows: List[pd.DataFrame] = []

    total_rows = 0
    seen_ref = 0
    seen_cand = 0

    for chunk in pd.read_csv(csv_path, chunksize=chunk_rows):
        if id_col not in chunk.columns:
            raise KeyError(f"ID column '{id_col}' not found in CSV.")
        total_rows += len(chunk)
        chunk["id_str"] = chunk[id_col].astype(str)

        # Add derived cols for consistency
        chunk = _add_derived_features(chunk)

        mask_ref = chunk["id_str"].isin(ref_set)
        mask_cand = chunk["id_str"].isin(cand_set)

        if mask_ref.any():
            sub = chunk.loc[mask_ref].copy()
            Z = _project_chunk(sub, meta["encoders"], proj, numeric_cols, onehot_levels, hash_dims)
            ref_embeddings.append(Z)
            seen_ref += len(sub)

        if mask_cand.any():
            sub = chunk.loc[mask_cand].copy()
            Z = _project_chunk(sub, meta["encoders"], proj, numeric_cols, onehot_levels, hash_dims)
            cand_embeddings.append(Z)
            cand_rows.append(sub)
            seen_cand += len(sub)

        if seen_ref >= len(ref_ids) and seen_cand >= len(cand_ids):
            break

    if seen_ref == 0 or seen_cand == 0:
        raise RuntimeError(f"No rows found for ref ({seen_ref}) or candidate ({seen_cand}). Check id column/splits.")

    Z_ref = np.vstack(ref_embeddings)
    Z_cand = np.vstack(cand_embeddings)
    df_cand = pd.concat(cand_rows, axis=0, ignore_index=True)

    print(f"[knn-ood] loaded rows: total={total_rows}, ref_used={len(Z_ref)}, cand_used={len(Z_cand)}")

    # Grid over k to pick the one that maximizes separation between ID-high and OOD-low quantiles
    gap_rows = []
    scores_per_k = {}
    for k in knn_k_list:
        nn = NearestNeighbors(n_neighbors=k, algorithm="auto", metric="euclidean")
        nn.fit(Z_ref)
        dists, _ = nn.kneighbors(Z_cand, n_neighbors=k, return_distance=True)
        score = dists.mean(axis=1)
        scores_per_k[k] = score
        id_hi_k = np.quantile(score, id_q[1])
        ood_lo_k = np.quantile(score, ood_q[0])
        gap = float(ood_lo_k - id_hi_k)
        gap_rows.append({"k": k, "gap": gap, "std": float(np.std(score))})

    gap_df = pd.DataFrame(gap_rows).sort_values("k")
    print("[knn-ood] k search (gap = q(ood_lo) - q(id_hi)):")
    print(gap_df.to_string(index=False))
    best_k = int(gap_df.loc[gap_df["gap"].idxmax(), "k"])
    print(f"[knn-ood] choosing k={best_k} (max gap)")

    knn_score = scores_per_k[best_k]

    id_lo, id_hi = np.quantile(knn_score, id_q[0]), np.quantile(knn_score, id_q[1])
    ood_lo, ood_hi = np.quantile(knn_score, ood_q[0]), np.quantile(knn_score, ood_q[1])

    print(f"[knn-ood] id range [{id_lo:.4f}, {id_hi:.4f}]")
    print(f"[knn-ood] ood range [{ood_lo:.4f}, {ood_hi:.4f}]")

    id_mask = (knn_score >= id_lo) & (knn_score <= id_hi)
    ood_mask = (knn_score >= ood_lo) & (knn_score <= ood_hi)

    df_cand["knn_score"] = knn_score
    df_id = df_cand.loc[id_mask].copy()
    df_ood = df_cand.loc[ood_mask].copy()

    out_dir.mkdir(parents=True, exist_ok=True)
    id_path = out_dir / f"{out_prefix}_id_knn.csv"
    ood_path = out_dir / f"{out_prefix}_ood_knn.csv"
    df_id.drop(columns=["id_str"], errors="ignore").to_csv(id_path, index=False)
    df_ood.drop(columns=["id_str"], errors="ignore").to_csv(ood_path, index=False)

    print(f"[knn-ood] wrote {len(df_id)} rows -> {id_path}")
    print(f"[knn-ood] wrote {len(df_ood)} rows -> {ood_path}")


def main():
    ap = argparse.ArgumentParser(description="Build kNN-based ID/OOD splits (single hard OOD).")
    ap.add_argument("--config", required=True, help="Path to YAML config.")
    ap.add_argument("--run-mode", choices=["local", "slurm"], default=None, help="Override run.mode in config.")
    ap.add_argument("--dry-run", action="store_true", help="If slurm: print sbatch but do not submit.")
    args = ap.parse_args()

    cfg = _load_config(Path(args.config))
    if args.run_mode:
        cfg.setdefault("run", {})["mode"] = args.run_mode

    # If run.mode == slurm, submit and exit.
    if _maybe_submit_slurm(cfg, args):
        return

    build_knn_splits(cfg)


if __name__ == "__main__":
    main()
