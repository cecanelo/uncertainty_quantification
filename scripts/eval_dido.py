#!/usr/bin/env python3
"""
Evaluate a trained DIDO AuxUE model and produce epistemic scores.
"""

from __future__ import annotations
import argparse
import csv
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import yaml

from data import DataConfig, _prepare_frame


def _load_preproc_meta(meta_path: Path) -> dict:
    meta = json.loads(Path(meta_path).read_text())
    if "encoders" not in meta:
        raise ValueError("preproc_meta.json missing 'encoders'.")
    return meta


def _build_features_from_meta(csv_path: Path, meta: dict) -> Tuple[np.ndarray, np.ndarray]:
    dc = DataConfig(
        csv_path=str(csv_path),
        target_col=meta.get("target_col", "price"),
        target_transform=meta.get("target", {}).get("mode", "none"),
    )
    df = _prepare_frame(dc)

    enc = meta["encoders"]
    numeric_cols = meta.get("numeric_cols", [])
    onehot_cols = meta.get("onehot_cols", [])
    hash_cols = meta.get("hash_cols", [])
    hash_dims = meta.get("hash_dims", {})

    feats: List[np.ndarray] = []

    # numeric
    for c in numeric_cols:
        stats = enc["num"][c]
        mean = float(stats["mean"])
        std = float(stats["std"]) if stats["std"] > 1e-12 else 1.0
        x = df[c].astype(float).to_numpy()
        feats.append(((x - mean) / std).reshape(-1, 1))

    # one-hot
    for c in onehot_cols:
        levels = enc["oh_levels"][c]
        s = df[c].astype(str).fillna("__NA__")
        mat = np.zeros((len(s), len(levels)), dtype=np.float32)
        idx = {lv: i for i, lv in enumerate(levels)}
        for i, val in enumerate(s):
            j = idx.get(val)
            if j is not None:
                mat[i, j] = 1.0
        feats.append(mat)

    # hashed
    import hashlib
    for c in hash_cols:
        n = int(hash_dims[c])
        s = df[c].astype(str).fillna("__NA__")
        mat = np.zeros((len(s), n), dtype=np.float32)
        for i, val in enumerate(s):
            h = hashlib.md5(f"{c}={val}".encode("utf-8")).hexdigest()
            j = int(h, 16) % n
            mat[i, j] = 1.0
        feats.append(mat)

    X = np.concatenate(feats, axis=1).astype(np.float32)

    y = df[dc.target_col].astype(float).to_numpy()
    tmode = meta.get("target", {}).get("mode", "none")
    if tmode == "log1p":
        y = np.log1p(y)

    return X, y.reshape(-1, 1).astype(np.float32)


def _read_base_preds(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "y_pred_det" not in df.columns:
        raise ValueError(f"{path} missing y_pred_det column.")
    if "id" not in df.columns and "row_index" not in df.columns and "row_idx" not in df.columns:
        raise ValueError(f"{path} must include an id/row_index/row_idx column.")
    return df


def _ensure_preds(split: str, cfg: dict, run_cfg: dict) -> Path:
    base_preds_cfg = run_cfg.get("base_preds", {}) or {}
    key = f"{split}_csv"
    if key in base_preds_cfg and base_preds_cfg[key]:
        p = Path(base_preds_cfg[key]).resolve()
        if p.exists():
            return p

    auto_create = bool(base_preds_cfg.get("auto_create", False))
    if not auto_create:
        raise FileNotFoundError(f"Base preds for split '{split}' not found and auto_create=False")

    base_run = run_cfg.get("base_run", {}) or {}
    base_cfg_path = Path(base_run.get("config_path", "")) if base_run.get("config_path") else None
    base_model_dir = Path(base_run.get("model_dir", "")) if base_run.get("model_dir") else None
    if base_cfg_path is None or base_model_dir is None:
        raise FileNotFoundError("base_run.config_path or base_run.model_dir missing for auto_create")

    base_cfg = yaml.safe_load(base_cfg_path.read_text())
    evals_root = base_cfg.get("io", {}).get("evals_root", "outputs/evals")
    # match eval_regression convention: use model_dir name as run tag
    run_tag = base_model_dir.name if base_model_dir is not None else base_cfg.get("slurm", {}).get("job_name", "run")

    cmd = [
        sys.executable,
        str(Path(__file__).resolve().parents[1] / "scripts" / "eval_regression.py"),
        "--config",
        str(base_cfg_path),
        "--outdir",
        str(base_model_dir),
        "--split",
        split,
    ]
    print(f"[dido-eval] Auto-creating base preds for split '{split}'")
    subprocess.run(cmd, check=True)
    fname = "test_preds.csv" if split == "test" else f"preds_{split}.csv"
    preds_path = Path(evals_root) / run_tag / fname
    if not preds_path.exists():
        raise FileNotFoundError(f"Expected preds at {preds_path} after auto-create.")
    return preds_path


class DirichletNet(torch.nn.Module):
    def __init__(self, in_dim: int, hidden: List[int], out_dim: int, activation: str = "relu", dropout: float = 0.0):
        super().__init__()
        act = torch.nn.ReLU if activation.lower() == "relu" else (torch.nn.GELU if activation.lower() == "gelu" else torch.nn.Tanh)
        layers: List[torch.nn.Module] = []
        prev = in_dim
        for h in hidden:
            layers.append(torch.nn.Linear(prev, h))
            layers.append(act())
            if dropout and dropout > 0:
                layers.append(torch.nn.Dropout(dropout))
            prev = h
        layers.append(torch.nn.Linear(prev, out_dim))
        self.net = torch.nn.Sequential(*layers)
        self.softplus = torch.nn.Softplus()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raw = self.net(x)
        return self.softplus(raw) + 1e-6


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dido-outdir", required=True, help="Directory containing trained DIDO model.pt")
    ap.add_argument("--split", default="test", choices=["train", "val", "test"], help="Which split to evaluate")
    args = ap.parse_args()

    dido_dir = Path(args.dido_outdir).resolve()
    cfg_path = dido_dir / "used_config.yaml"
    run_meta_path = dido_dir / "run_meta.json"
    model_path = dido_dir / "model.pt"

    if not cfg_path.exists() or not model_path.exists():
        raise SystemExit("DIDO run is missing used_config.yaml or model.pt")

    cfg = yaml.safe_load(cfg_path.read_text())
    run_meta = json.loads(run_meta_path.read_text()) if run_meta_path.exists() else {}

    bin_cfg = run_meta.get("binning", cfg.get("binning", {}))
    K = int(bin_cfg.get("K", 10))
    include_mu = bool(bin_cfg.get("include_mu_as_feature", cfg.get("binning", {}).get("include_mu_as_feature", True)))

    base_meta_path = Path(cfg["base_artifacts"]["preproc_meta_path"]).resolve()
    meta = _load_preproc_meta(base_meta_path)
    X_all, _ = _build_features_from_meta(Path(cfg["data"]["csv_path"]).resolve(), meta)

    # Build mapping from external IDs to row indices so that we can safely
    # index into X_all even when IDs are not 0..N-1.
    id_to_row_index = None
    if "id_values" in meta:
        id_values = np.asarray(meta["id_values"], dtype=int)
        id_to_row_index = {int(v): int(i) for i, v in enumerate(id_values)}

    preds_path = _ensure_preds(args.split, cfg, cfg)
    preds_df = _read_base_preds(preds_path)

    def _get_ids(df: pd.DataFrame) -> np.ndarray:
        """
        Return row indices into X_all for the given predictions frame.
        Prefer explicit row index columns, but fall back to mapping external IDs
        via preproc_meta.id_values when only an 'id' column is present.
        """
        if "row_idx" in df.columns:
            return df["row_idx"].to_numpy(dtype=int)
        if "row_index" in df.columns:
            return df["row_index"].to_numpy(dtype=int)
        if "id" in df.columns:
            if id_to_row_index is None:
                raise ValueError(
                    "Base prediction file contains 'id' but preproc_meta.json is "
                    "missing 'id_values'; cannot map IDs to row indices."
                )
            ids = df["id"].to_numpy()
            row_indices: List[int] = []
            for raw_id in ids:
                key = int(raw_id)
                if key not in id_to_row_index:
                    raise IndexError(
                        f"ID {key} from base predictions not found in preproc_meta.id_values."
                    )
                row_indices.append(id_to_row_index[key])
            return np.asarray(row_indices, dtype=int)
        raise ValueError(
            "Base prediction file must contain one of 'row_idx', 'row_index', or 'id' columns."
        )

    ids = _get_ids(preds_df).astype(int)
    # If indices look out of bounds but we have id mapping, try a remap as a safeguard.
    if ids.max() >= X_all.shape[0] and id_to_row_index is not None and "id" in preds_df.columns:
        remapped = []
        for raw_id in preds_df["id"].to_numpy():
            key = int(raw_id)
            if key not in id_to_row_index:
                raise IndexError(f"ID {key} from base predictions not found in preproc_meta.id_values.")
            remapped.append(id_to_row_index[key])
        ids = np.asarray(remapped, dtype=int)
        if ids.max() >= X_all.shape[0]:
            raise IndexError(f"Remapped row indices still out of bounds (max {ids.max()} vs X dim {X_all.shape[0]}).")
    mu_pred = preds_df["y_pred_det"].to_numpy()

    X_split = X_all[ids]
    if include_mu:
        X_split = np.concatenate([X_split, mu_pred.reshape(-1, 1).astype(np.float32)], axis=1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_cfg = cfg.get("model", {})
    hidden = model_cfg.get("hidden_dims", [256, 128])
    act = model_cfg.get("activation", "relu")
    dropout = float(model_cfg.get("dropout", 0.0))
    model = DirichletNet(X_split.shape[1], hidden, K, activation=act, dropout=dropout).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    with torch.no_grad():
        alpha = model(torch.tensor(X_split, dtype=torch.float32, device=device))
        strength = alpha.sum(dim=1)
        mean_probs = (alpha / strength[:, None])
        # move to CPU before numpy
        strength_np = strength.cpu().numpy()
        mean_probs_np = mean_probs.cpu().numpy()
        entropy = -np.sum(mean_probs_np * np.log(mean_probs_np + 1e-12), axis=1)
        vacuity = (K / strength_np)

    out_root = Path(cfg.get("io", {}).get("evals_root", "outputs/evals"))
    # Prefer the actual run directory name so the eval folder carries the job id/ts
    run_tag = dido_dir.name
    if not run_tag:
        job_name = cfg.get("slurm", {}).get("job_name", "dido")
        run_tag = job_name if str(job_name).startswith("dido") else f"dido_{job_name}"
    eval_dir = out_root / run_tag
    eval_dir.mkdir(parents=True, exist_ok=True)
    preds_out = eval_dir / f"dido_{args.split}.csv"

    with preds_out.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "split", "dido_strength_raw", "dido_vacuity_raw", "dido_entropy_raw"])
        for rid, s, v, e in zip(ids, strength, vacuity, entropy):
            w.writerow([rid, args.split, float(s), float(v), float(e)])

    metrics = {
        "split": args.split,
        "mean_strength": float(np.mean(strength_np)),
        "mean_vacuity": float(np.mean(vacuity)),
        "mean_entropy": float(np.mean(entropy)),
    }
    (eval_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))

    print(f"[dido-eval] Saved scores to: {preds_out}")


if __name__ == "__main__":
    main()
