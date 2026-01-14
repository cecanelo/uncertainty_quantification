#!/usr/bin/env python3
"""
Evaluate a trained normalizing flow (NF) on an OOD CSV, using the same
base model, preprocessing, and NF hyperparameters as in the ID setting.

Typical usage (single OOD set):

    python scripts/eval_flows_ood.py \\
        --config outputs/trainings/training_nf_lpl_m/nf_lpl_m_.../used_config.yaml \\
        --outdir outputs/trainings/training_nf_lpl_m/nf_lpl_m_... \\
        --ood-csv datasets/craigslist_ood_geo_fl_tx.csv \\
        --dataset-label geo

This script:
  * Loads the NF config + checkpoint from `--config` / `--outdir`.
  * Uses the same base_run and base_artifacts.preproc_meta_path as ID eval.
  * Rebuilds features for the OOD CSV with the base encoders.
  * Rebuilds the base regression head, runs it in transformed space to get
    y_true_t, mu_t, and scale_t (std-like) exactly as eval_regression does.
  * Forms z = (y_true_t - mu_t) / scale_t and evaluates log p(z | x) under
    the trained NF.
  * Writes per-row outputs under evals_root/run_tag as:
        flow_eval_ood_<label>.csv
    with the same columns as flow_eval_{split}.csv used for ID.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Tuple
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import yaml

from data import DataConfig, _prepare_frame, _apply_encoders, _target_transform, inverse_target
from model_base import MLPRegressor, gaussian_nll, laplace_nll
from train_regression import _load_cfg as _load_base_cfg
from train_flows import _build_flow, _load_cfg as _load_nf_cfg
try:
    from .config_resolver import load_and_resolve_config  # type: ignore
except Exception:
    from config_resolver import load_and_resolve_config  # type: ignore


def _load_preproc_meta(meta_path: Path) -> dict:
    meta = json.loads(meta_path.read_text())
    if "encoders" not in meta:
        raise ValueError("preproc_meta.json missing 'encoders'.")
    return meta


def _delta_sigma_orig(
    head_type: str,
    sigma_z: np.ndarray | None,
    mu_orig: np.ndarray,
    target_meta: Dict[str, str],
) -> np.ndarray:
    """
    Map transformed-space std sigma_z to approximate original-space std,
    mirroring eval_regression/_delta_sigma_orig.
    """
    n = mu_orig.shape[0]
    if sigma_z is None:
        return np.full(n, np.nan, dtype=float)

    sigma_z = sigma_z.reshape(-1)
    if head_type == "point":
        return np.full(n, np.nan, dtype=float)

    mode = (target_meta or {}).get("mode", "none").lower()
    if mode == "log1p":
        # dy/dz = exp(z) ≈ y + 1, approximate with predicted y
        return (mu_orig + 1.0) * sigma_z
    return sigma_z


def _set_seed(seed: int | None) -> None:
    if seed is None:
        return
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _sample_z_std(
    flow: torch.nn.Module,
    X: np.ndarray,
    *,
    n_samples: int,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    if n_samples <= 0:
        return np.full((X.shape[0],), np.nan, dtype=np.float32)
    n = X.shape[0]
    out = np.empty(n, dtype=np.float32)
    for i in range(0, n, batch_size):
        xb = torch.tensor(X[i : i + batch_size], dtype=torch.float32, device=device)
        samples = flow.sample(n_samples, context=xb)
        if samples.dim() == 3:
            if samples.shape[0] == n_samples:
                std = samples.std(dim=0, unbiased=False)
            elif samples.shape[1] == n_samples:
                std = samples.std(dim=1, unbiased=False)
            else:
                std = samples.std(dim=0, unbiased=False)
        else:
            std = torch.zeros((samples.shape[0], 1), device=samples.device)
        std = std.squeeze(-1).detach().cpu().numpy()
        out[i : i + len(std)] = std
    return out


def _build_features_ood(csv_path: Path, meta: dict) -> Tuple[np.ndarray, np.ndarray, Dict[str, str], pd.DataFrame]:
    """
    Build OOD features and transformed targets using the same encoders and
    target transform as the base run.
    """
    dc = DataConfig(
        csv_path=str(csv_path),
        target_col=meta.get("target_col", "price"),
        target_transform=meta.get("target", {}).get("mode", "none"),
    )
    df = _prepare_frame(dc)

    y = df[dc.target_col].astype(float).to_numpy()
    tmode = meta.get("target", {}).get("mode", "none")
    y_tr, y_meta = _target_transform(y, tmode)

    numeric_cols = meta.get("numeric_cols", [])
    onehot_cols = meta.get("onehot_cols", [])
    hash_cols = meta.get("hash_cols", [])
    enc = meta["encoders"]

    X = _apply_encoders(df, numeric_cols, onehot_cols, hash_cols, enc)
    return X.astype(np.float32), y_tr.reshape(-1, 1).astype(np.float32), y_meta, df


def _eval_base_on_ood(
    base_cfg_path: Path,
    base_outdir: Path,
    meta: dict,
    X_ood: np.ndarray,
    y_tr_ood: np.ndarray,
) -> Tuple[str, np.ndarray, np.ndarray]:
    """
    Run the base regression head on OOD features/targets in transformed space.
    Returns (head_type, mu_t, scale_t_std) where scale_t_std is a std-like
    scale in transformed space (for Laplace we convert b_z -> std_z).
    """
    base_cfg = _load_base_cfg(str(base_cfg_path))
    model_cfg = base_cfg["model"]
    head_type = str(model_cfg.get("head_type", "point")).lower()

    in_dim = int(meta["feature_dim"])
    hidden = model_cfg.get("hidden_dims", [512, 256, 128])
    device = torch.device("cpu")

    model = MLPRegressor(
        in_dim=in_dim,
        hidden_dims=hidden,
        head_type=head_type,
        activation=model_cfg.get("activation", "relu"),
        dropout=float(model_cfg.get("dropout", 0.1)),
        use_batchnorm=False,
    ).to(device)

    ckpt_path = base_outdir / "model.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Base checkpoint model.pt not found in {base_outdir}")
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state["model_state_dict"])
    model.eval()

    xb = torch.tensor(X_ood, dtype=torch.float32, device=device)
    yb = torch.tensor(y_tr_ood, dtype=torch.float32, device=device)

    with torch.no_grad():
        out = model(xb)
        mu = out["mu"]
        mu_np = mu.cpu().numpy().reshape(-1)
        sigma_z = None
        if head_type == "gauss":
            sigma = out["sigma"]
            # nll for sanity, though we don't use it here
            _ = gaussian_nll(mu, sigma, yb)
            sigma_z = sigma.cpu().numpy().reshape(-1)
        elif head_type == "laplace":
            b = out["b"]
            _ = laplace_nll(mu, b, yb)
            # convert Laplace scale b_z to Gaussian-equivalent std in z-space
            sigma_z = (np.sqrt(2.0) * b.cpu().numpy().reshape(-1))

    return head_type, mu_np, sigma_z


def submit_slurm(
    cfg_path: Path,
    outdir: Path,
    ood_csv: str | None,
    ood_dir: str | None,
    dataset_label: str | None,
    n_samples: int | None,
    seed: int | None,
    cfg: dict,
) -> None:
    slurm_cfg = cfg.get("slurm", {}) or {}
    partition = slurm_cfg.get("partition", "TEST")
    time_str = slurm_cfg.get("time", "00:30:00")
    mem_gb = int(slurm_cfg.get("mem_gb", 16))
    cpus = int(slurm_cfg.get("cpus", 2))
    gpus = int(slurm_cfg.get("gpus", 1))
    conda_env = slurm_cfg.get("conda_env", "thesis")
    job_name = slurm_cfg.get("job_name", "eval_flows_ood")

    logs_root = (Path(__file__).resolve().parents[1] / "logs").resolve()
    logs_root.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    out_log = logs_root / f"{job_name}_flow_eval_{ts}_%j.out"
    err_log = logs_root / f"{job_name}_flow_eval_{ts}_%j.err"

    script_lines = [
        "#!/bin/bash",
        f"#SBATCH --job-name={job_name}_eval",
        f"#SBATCH --partition={partition}",
        f"#SBATCH --cpus-per-task={cpus}",
        f"#SBATCH --mem={mem_gb}G",
        f"#SBATCH --time={time_str}",
        f"#SBATCH --output={out_log}",
        f"#SBATCH --error={err_log}",
    ]
    if gpus > 0:
        script_lines.insert(6, f"#SBATCH --gres=gpu:{gpus}")

    cmd = [
        "python",
        str(Path(__file__).resolve()),
        "--config",
        str(cfg_path),
        "--outdir",
        str(outdir),
        "--mode",
        "local",
    ]
    if ood_csv:
        cmd += ["--ood-csv", ood_csv]
    if ood_dir:
        cmd += ["--ood-dir", ood_dir]
    if dataset_label:
        cmd += ["--dataset-label", dataset_label]
    if n_samples is not None:
        cmd += ["--n-samples", str(n_samples)]
    if seed is not None:
        cmd += ["--seed", str(seed)]

    script_lines += [
        f"cd \"{Path(__file__).resolve().parents[1]}\"",
        'source "$HOME/miniconda3/etc/profile.d/conda.sh"',
        f"conda activate {conda_env}",
        'echo "[env] host=$(hostname) date=$(date)"',
        " ".join(cmd),
    ]

    sb_script = "\n".join(script_lines) + "\n"
    print("[eval-flows-ood][slurm] sbatch script:\n")
    print(sb_script)

    res = subprocess.run(["sbatch"], input=sb_script.encode("utf-8"), check=False, capture_output=True)
    if res.returncode != 0:
        print(res.stdout.decode())
        print(res.stderr.decode(), file=sys.stderr)
        res.check_returncode()
    else:
        print(res.stdout.decode().strip())


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate an NF model on an OOD CSV.")
    ap.add_argument("--config", required=True, help="NF used_config.yaml (from train_flows)")
    ap.add_argument("--outdir", required=True, help="NF run directory (contains model.pt)")
    group = ap.add_mutually_exclusive_group(required=True)
    group.add_argument("--ood-csv", help="Single OOD CSV to evaluate")
    group.add_argument("--ood-dir", help="Directory with multiple *_ood_*.csv files")
    ap.add_argument("--dataset-label", help="Label to use for this OOD set (only with --ood-csv)")
    ap.add_argument("--n-samples", type=int, default=None, help="Flow samples per row for sigma estimate (default: cfg nf_eval.n_samples or 200)")
    ap.add_argument("--seed", type=int, default=None, help="Seed for flow sampling (default: cfg nf_eval.seed or data.split_seed)")
    ap.add_argument("--mode", choices=["local", "slurm"], default="local", help="Run locally or submit a slurm job")
    args = ap.parse_args()

    cfg_path = Path(args.config).resolve()
    nf_outdir = Path(args.outdir).resolve()
    if not cfg_path.exists():
        raise FileNotFoundError(f"NF config not found: {cfg_path}")
    if not nf_outdir.exists():
        raise FileNotFoundError(f"NF outdir not found: {nf_outdir}")
    if not (nf_outdir / "model.pt").exists():
        raise FileNotFoundError(f"NF checkpoint model.pt not found in {nf_outdir}")

    cfg = _load_nf_cfg(cfg_path)
    if args.mode == "slurm":
        submit_slurm(cfg_path, nf_outdir, args.ood_csv, args.ood_dir, args.dataset_label, args.n_samples, args.seed, cfg)
        return

    # Resolve base artifacts and run paths as in train_flows/eval_flows
    base_run_cfg = cfg.get("base_run", {}) or {}
    base_cfg_path = Path(base_run_cfg.get("config_path", "")).expanduser().resolve()
    base_model_dir = Path(base_run_cfg.get("model_dir", "")).expanduser().resolve()
    if not base_cfg_path.exists() or not base_model_dir.exists():
        raise SystemExit("NF config base_run.config_path or base_run.model_dir is missing/invalid.")

    meta_path = Path(cfg["base_artifacts"]["preproc_meta_path"]).expanduser().resolve()
    if not meta_path.exists():
        raise FileNotFoundError(f"base_artifacts.preproc_meta_path not found: {meta_path}")
    meta = _load_preproc_meta(meta_path)

    # Discover OOD CSVs
    ood_specs = []
    if args.ood_csv:
        ood_csv = Path(args.ood_csv).resolve()
        if not ood_csv.exists():
            raise FileNotFoundError(f"OOD CSV not found: {ood_csv}")
        label = args.dataset_label or ood_csv.stem
        ood_specs.append((ood_csv, label))
    else:
        root = Path(args.ood_dir).resolve()
        if not root.exists():
            raise FileNotFoundError(f"OOD dir not found: {root}")
        for p in sorted(root.glob("*.csv")):
            if "_ood_" not in p.name:
                continue
            stem = p.stem
            label = stem.split("_ood_", 1)[1] if "_ood_" in stem else stem
            ood_specs.append((p.resolve(), label))
        if not ood_specs:
            raise SystemExit(f"No *_ood_*.csv files found under {root}")

    # NF construction (same hyperparameters as training)
    nf_cfg = cfg.get("nf", {}) or {}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for ood_csv, label in ood_specs:
        # 1) Build features and transformed targets for this OOD CSV
        X_ood, y_tr_ood, y_meta, df_ood = _build_features_ood(ood_csv, meta)

        # 2) Evaluate the base regression head on OOD (transformed space)
        head_type, mu_t, sigma_t = _eval_base_on_ood(
            base_cfg_path=base_cfg_path,
            base_outdir=base_model_dir,
            meta=meta,
            X_ood=X_ood,
            y_tr_ood=y_tr_ood,
        )

        if sigma_t is None or head_type not in ("gauss", "laplace"):
            raise SystemExit(
                f"NF OOD eval expects a probabilistic base head (gauss/laplace); got head_type={head_type}"
            )

        # 3) Compute standardized residuals z
        z = (y_tr_ood.reshape(-1) - mu_t) / np.maximum(sigma_t, 1e-8)

        # 4) Build NF and load checkpoint
        cond_dim = int(X_ood.shape[1])
        flow = _build_flow(
            cond_dim=cond_dim,
            transform=nf_cfg.get("transform", "affine"),
            hidden_features=int(nf_cfg.get("hidden_features", 256)),
            num_layers=int(nf_cfg.get("num_layers", 6)),
            actnorm=bool(nf_cfg.get("actnorm", True)),
            num_bins=int(nf_cfg.get("num_bins", 8)),
        ).to(device)
        state = torch.load(nf_outdir / "model.pt", map_location=device)
        flow.load_state_dict(state["flow_state_dict"])
        flow.eval()

        with torch.no_grad():
            xb = torch.tensor(X_ood, dtype=torch.float32, device=device)
            zb = torch.tensor(z.reshape(-1, 1).astype(np.float32), dtype=torch.float32, device=device)
            log_prob = flow.log_prob(inputs=zb, context=xb)  # [N]
            log_prob_np = log_prob.cpu().numpy().reshape(-1)
        mean_nll = float(-log_prob_np.mean())

        # 5) Map predictions back to original space and approximate sigma in original units
        y_true_orig = inverse_target(y_tr_ood.reshape(-1, 1), y_meta).reshape(-1)
        mu_orig = inverse_target(mu_t.reshape(-1, 1), y_meta).reshape(-1)
        sigma_ale_orig = _delta_sigma_orig(head_type, sigma_t, mu_orig, y_meta)
        nf_eval_cfg = cfg.get("nf_eval", {}) or cfg.get("evaluation", {}) or {}
        n_samples = int(args.n_samples or nf_eval_cfg.get("n_samples", nf_eval_cfg.get("n_nf_samples", 200)))
        seed = args.seed
        if seed is None:
            seed = nf_eval_cfg.get("seed")
        if seed is None:
            seed = (cfg.get("data", {}) or {}).get("split_seed")
        _set_seed(None if seed is None else int(seed))
        batch_size = int(nf_eval_cfg.get("batch_size", cfg.get("training", {}).get("batch_size", 1024)))
        batch_size = max(1, batch_size)
        z_std = _sample_z_std(flow, X_ood, n_samples=n_samples, batch_size=batch_size, device=device)
        sigma_nf_t = sigma_t * z_std
        sigma_nf_orig = _delta_sigma_orig(head_type, sigma_nf_t, mu_orig, y_meta)

        # 6) Write per-row outputs under evals_root/run_tag
        eval_cfg = cfg.get("evaluation", {}) or {}
        evals_root = eval_cfg.get("evals_root", cfg.get("io", {}).get("evals_root", "outputs/evals"))
        run_tag = nf_outdir.name
        save_dir = Path(evals_root).expanduser().resolve() / run_tag
        save_dir.mkdir(parents=True, exist_ok=True)
        out_csv = save_dir / f"flow_eval_ood_{label}.csv"

        # IDs: prefer 'id' column from OOD CSV if present, else row index
        if "id" in df_ood.columns:
            ids = df_ood["id"].to_numpy()
        else:
            ids = np.arange(len(df_ood), dtype=int)

        with out_csv.open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "id",
                    "split",
                    "head_type",
                    "y_true_orig",
                    "y_pred_base_orig",
                    "sigma_base_orig",
                    "sigma_nf_orig",
                    "z_std",
                    "z_raw",
                    "log_prob_z_raw",
                ]
            )
            for rid, yt_o, mu_o, sig_o, sig_nf_o, z_s, z_i, lp in zip(
                ids, y_true_orig, mu_orig, sigma_ale_orig, sigma_nf_orig, z_std, z, log_prob_np
            ):
                writer.writerow(
                    [
                        int(rid),
                        "test",  # treat OOD as test-like split
                        head_type,
                        float(yt_o),
                        float(mu_o),
                        float(sig_o) if np.isfinite(sig_o) else np.nan,
                        float(sig_nf_o) if np.isfinite(sig_nf_o) else np.nan,
                        float(z_s) if np.isfinite(z_s) else np.nan,
                        float(z_i),
                        float(lp),
                    ]
                )

        metrics = {
            "split": "test",
            "dataset": label,
            "n": int(len(ids)),
            "mean_nll": mean_nll,
            "head_type": head_type,
            "n_samples": int(n_samples),
            "seed": None if seed is None else int(seed),
        }
        with (save_dir / f"metrics_ood_{label}.json").open("w") as f:
            json.dump(metrics, f, indent=2)

        print(
            f"[eval_flows_ood] csv={ood_csv.name} label={label} n={len(ids)} "
            f"head_type={head_type} mean_nll={mean_nll:.4f}"
        )
        print(f"[eval_flows_ood] wrote rows to: {out_csv}")


if __name__ == "__main__":
    main()
