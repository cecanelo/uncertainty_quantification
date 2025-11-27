# uncertainty_quantification/scripts/train_flows.py
from __future__ import annotations
import argparse, csv, json, subprocess, sys, time
from pathlib import Path
from typing import Any, Dict, Tuple
from datetime import datetime
import os

import numpy as np
import pandas as pd
import torch
import torch.optim as optim
import yaml
from data import DataConfig, _prepare_frame

REPO_ROOT = Path(__file__).resolve().parents[1]

# ----------------------------- helpers -----------------------------
def _infer_head_type(cfg: dict) -> str:
    """
    Try to infer base head_type for naming. Prefer base_run.config_path -> model.head_type.
    Fallbacks: cfg['head_type'] or 'nf'.
    """
    # From base_run config
    base_run = cfg.get("base_run", {}) or {}
    cfg_path = base_run.get("config_path")
    if cfg_path:
        try:
            with Path(cfg_path).open("r") as f:
                base_cfg = yaml.safe_load(f)
            ht = base_cfg.get("model", {}).get("head_type")
            if ht:
                return str(ht).lower()
        except Exception:
            pass
    # Fallbacks
    ht_cfg = cfg.get("head_type")
    if ht_cfg:
        return str(ht_cfg).lower()
    return "nf"

# ----------------------------- util -----------------------------
def _load_cfg(p: str | Path) -> Dict[str, Any]:
    with open(p, "r") as f:
        return yaml.safe_load(f)

def _save_yaml(obj: Dict[str, Any], path: Path) -> None:
    with open(path, "w") as f:
        yaml.safe_dump(obj, f, sort_keys=False)

# ----------------------- flow construction ----------------------
def _build_flow(cond_dim: int,
                transform: str = "affine",
                hidden_features: int = 256,
                num_layers: int = 6,
                actnorm: bool = True,
                num_bins: int = 8):
    """
    Conditional flow p(z | x) for scalar z with x as context.
    - transform: "affine" (MAF) or "spline" (RQs spline-AR)
    - features=1 because z is 1D
    Notes:
      * No permutations are inserted since D=1.
      * ActNorm can stabilize deeper stacks.
    """
    try:
        from nflows.flows import Flow
        from nflows.distributions import StandardNormal
        from nflows.transforms import CompositeTransform
        from nflows.transforms.autoregressive import (
            MaskedAffineAutoregressiveTransform,
        )
        from nflows.transforms.normalization import ActNorm as NFActNorm
    except ImportError as e:
        raise SystemExit(
            f"train_flows.py needs `nflows`. Install it, e.g.: pip install nflows. (ImportError: {e})"
        ) from e

    layers = []
    for _ in range(num_layers):
        if transform.lower() == "spline":
            try:
                from nflows.transforms.spline import (
                    MaskedPiecewiseRationalQuadraticAutoregressiveTransform as SplineAR,
                )
            except ImportError as e:
                raise SystemExit(
                    "Spline transform requested but nflows spline module is missing. "
                    "Install from source: pip install git+https://github.com/bayesiains/nflows"
                ) from e
            layers.append(
                SplineAR(
                    features=1,
                    hidden_features=hidden_features,
                    context_features=cond_dim,
                    num_bins=num_bins,
                    tails="linear",
                )
            )
        else:
            layers.append(
                MaskedAffineAutoregressiveTransform(
                    features=1,
                    hidden_features=hidden_features,
                    context_features=cond_dim,
                )
            )
        if actnorm:
            layers.append(NFActNorm(features=1))

    transform = CompositeTransform(layers)
    base = StandardNormal([1])
    return Flow(transform, base)

# ---------------------- post-hoc data helpers -------------------
def _load_preproc_meta(meta_path: Path) -> dict:
    meta = json.loads(Path(meta_path).read_text())
    if "encoders" not in meta:
        raise ValueError("preproc_meta.json missing 'encoders'.")
    return meta

def _to_float(s):
    try:
        return float(s)
    except Exception:
        return np.nan

def _build_features_from_meta(csv_path: Path, meta: dict) -> Tuple[np.ndarray, np.ndarray]:
    # Reuse base preprocessing (adds derived columns) via DataConfig/_prepare_frame
    dc = DataConfig(
        csv_path=str(csv_path),
        target_col=meta.get("target_col", "price"),
        target_transform=meta.get("target", {}).get("mode", "none"),
    )
    df = _prepare_frame(dc)

    enc = meta["encoders"]
    numeric_cols = meta.get("numeric_cols", [])
    onehot_cols  = meta.get("onehot_cols", [])
    hash_cols    = meta.get("hash_cols", [])
    hash_dims    = meta.get("hash_dims", {})

    feats = []

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

    # hashed high-card columns
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

    # target transform (consistent with base run)
    y = df[dc.target_col].astype(float).to_numpy()
    tmode = meta.get("target", {}).get("mode", "none")
    if tmode == "log1p":
        y = np.log1p(y)

    return X, y.reshape(-1, 1).astype(np.float32)

def _read_preds(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"y_true_t", "mu_t", "scale_t"}
    if not required.issubset(df.columns):
        raise ValueError(f"{path} must include columns {required} plus an id/row_idx column.")
    if "id" not in df.columns and "row_idx" not in df.columns:
        raise ValueError(f"{path} must include 'id' or 'row_idx' column.")
    return df

def _build_split_tensors(X_all: np.ndarray,
                         df_preds: pd.DataFrame,
                         standardize: bool) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Build tensors for a split:
      X_split: features from base run encoders
      z_split: standardized residuals (z = (y_true_t - mu_t) / scale_t)
    """
    idx_col = "id" if "id" in df_preds.columns else "row_idx"
    idx = df_preds[idx_col].astype(int).to_numpy()

    y_true_t = df_preds["y_true_t"].to_numpy()
    mu_t = df_preds["mu_t"].to_numpy()
    scale_t = df_preds["scale_t"].to_numpy()

    z = y_true_t - mu_t
    if standardize:
        z = z / np.maximum(scale_t, 1e-8)

    Xs = torch.tensor(X_all[idx], dtype=torch.float32)
    zs = torch.tensor(z.reshape(-1, 1).astype(np.float32), dtype=torch.float32)
    return Xs, zs

# ----------------------------- train ----------------------------
def submit_slurm(cfg_path: Path, outdir_raw: str, cfg: dict) -> None:
    head_type = _infer_head_type(cfg)
    job_name_cfg = cfg.get("slurm", {}).get("job_name", "nf")

    slurm_cfg = cfg.get("slurm", {}) or {}
    partition = slurm_cfg.get("partition", "TEST")
    time_str = slurm_cfg.get("time", "00:30:00")
    mem_gb = int(slurm_cfg.get("mem_gb", 16))
    cpus = int(slurm_cfg.get("cpus", 2))
    gpus = int(slurm_cfg.get("gpus", 0))
    conda_env = slurm_cfg.get("conda_env", "thesis")
    job_name = slurm_cfg.get("job_name", "train_flows")

    logs_root = (REPO_ROOT / "logs").resolve()
    logs_root.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    out_log = logs_root / f"{job_name}_{ts}_%j.out"
    err_log = logs_root / f"{job_name}_{ts}_%j.err"

    script_lines = [
        "#!/bin/bash",
        f"#SBATCH --job-name={job_name}",
        f"#SBATCH --partition={partition}",
        f"#SBATCH --cpus-per-task={cpus}",
        f"#SBATCH --mem={mem_gb}G",
        f"#SBATCH --time={time_str}",
        f"#SBATCH --output={out_log}",
        f"#SBATCH --error={err_log}",
    ]
    if gpus > 0:
        script_lines.insert(6, f"#SBATCH --gres=gpu:{gpus}")

    # Fill outdir with head/ts placeholders; leave {jobid} to be resolved on the worker
    outdir_str = str(outdir_raw)
    outdir_str = (
        outdir_str
        .replace("{head}", head_type)
        .replace("{ts}", ts)
        .replace("{job}", job_name_cfg)
    )

    script_lines += [
        f"cd \"{REPO_ROOT}\"",
        'source "$HOME/miniconda3/etc/profile.d/conda.sh"',
        f"conda activate {conda_env}",
        'echo "[env] host=$(hostname) date=$(date)"',
        f"python \"{REPO_ROOT / 'scripts' / 'train_flows.py'}\" --config \"{cfg_path}\" --outdir \"{outdir_str}\" --mode local",
    ]
    sb_script = "\n".join(script_lines) + "\n"
    print("[train_flows][slurm] sbatch script:\n")
    print(sb_script)

    res = subprocess.run(["sbatch"], input=sb_script.encode("utf-8"), check=False, capture_output=True)
    if res.returncode != 0:
        print(res.stdout.decode())
        print(res.stderr.decode(), file=sys.stderr)
        res.check_returncode()
    else:
        print(res.stdout.decode().strip())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="YAML config path")
    ap.add_argument("--outdir", required=False, help="Output directory (default: io.outdir from config)")
    ap.add_argument("--mode", choices=["local", "slurm"], default="local", help="local or slurm submission")
    args = ap.parse_args()

    cfg = _load_cfg(args.config)
    cfg_outdir = cfg.get("io", {}).get("outdir")

    def _resolve_outdir(raw: str) -> Path:
        ts = datetime.now().strftime("%Y%m%d-%H%M%S")
        head_type = _infer_head_type(cfg)
        job_name_cfg = cfg.get("slurm", {}).get("job_name", "nf")
        jobid_env = os.environ.get("SLURM_JOB_ID", "NA")
        filled = (
            raw
            .replace("{head}", head_type)
            .replace("{ts}", ts)
            .replace("{jobid}", jobid_env)
            .replace("{job}", job_name_cfg)
        )
        return Path(filled).resolve()

    if args.mode == "slurm":
        raw_outdir = args.outdir or cfg_outdir
        if not raw_outdir:
            raise SystemExit("Please specify an outdir via --outdir or io.outdir in the config.")
        submit_slurm(Path(args.config).resolve(), raw_outdir, cfg)
        return

    outdir = None
    if args.outdir:
        outdir = _resolve_outdir(str(args.outdir))
    elif cfg_outdir:
        outdir = _resolve_outdir(str(cfg_outdir))
    if outdir is None:
        raise SystemExit("Please specify an outdir via --outdir or io.outdir in the config.")
    outdir.mkdir(parents=True, exist_ok=True)

    seed = int(cfg.get("seed", 42))
    torch.manual_seed(seed)
    np.random.seed(seed)
    start_time = time.perf_counter()
    start_utc = datetime.utcnow().isoformat() + "Z"

    device_str = str(cfg.get("training", {}).get("device", "cuda")).lower()
    device = torch.device("cuda" if (torch.cuda.is_available() and device_str == "cuda") else "cpu")

    # --- Rebuild features with base encoders, load predictions per split
    base_meta_path = Path(cfg["base_artifacts"]["preproc_meta_path"]).resolve()
    meta = _load_preproc_meta(base_meta_path)

    X_all, _ = _build_features_from_meta(Path(cfg["data"]["csv_path"]).resolve(), meta)
    cond_dim = int(X_all.shape[1])

    standardize = bool(cfg["nf"].get("standardize", True))

    base_preds_cfg = cfg.get("base_preds", {}) or {}
    flow_cfg = cfg.get("flow_dumps", {}) or {}
    auto_create = bool(flow_cfg.get("auto_create", False))
    flow_outdir = Path(flow_cfg.get("output_dir") or outdir).resolve()
    flow_outdir.mkdir(parents=True, exist_ok=True)
    base_run_cfg = cfg.get("base_run", {}) or {}
    base_cfg_path = Path(base_run_cfg.get("config_path", "")) if base_run_cfg.get("config_path") else None
    base_model_dir = Path(base_run_cfg.get("model_dir", "")) if base_run_cfg.get("model_dir") else None
    flow_created = []

def _ensure_flow(split: str) -> Path:
        key = f"{split}_csv"
        if key in base_preds_cfg:
            p = Path(base_preds_cfg[key]).resolve()
            if p.exists():
                return p
        if not auto_create:
            raise FileNotFoundError(f"Flow CSV for split '{split}' not found and auto_create=False")
        if base_cfg_path is None or base_model_dir is None:
            raise FileNotFoundError("base_run.config_path or base_run.model_dir missing/invalid for auto_create")
        if not base_cfg_path.exists() or not base_model_dir.exists():
            raise FileNotFoundError("base_run.config_path or base_run.model_dir missing/invalid for auto_create")
        cmd = [
            sys.executable,
            str(REPO_ROOT / "scripts" / "eval_regression.py"),
            "--config",
            str(base_cfg_path),
            "--outdir",
            str(base_model_dir),
            "--split",
            split,
            "--flow-dump",
            "--flow-outdir",
            str(flow_outdir),
        ]
        print(f"[nf] Auto-creating flow CSV for split '{split}' via eval_regression.py")
        subprocess.run(cmd, check=True)
        p = flow_outdir / f"flow_{split}.csv"
        if not p.exists():
            raise FileNotFoundError(f"Expected flow dump not found at {p}")
        flow_created.append({"split": split, "path": str(p)})
        return p

    preds_train_path = _ensure_flow("train")
    preds_val_path = _ensure_flow("val")

    preds_train = _read_preds(preds_train_path)
    preds_val   = _read_preds(preds_val_path)

    X_tr, z_tr = _build_split_tensors(X_all, preds_train, standardize)
    X_va, z_va = _build_split_tensors(X_all, preds_val,   standardize)

    from torch.utils.data import TensorDataset, DataLoader
    bs = int(cfg.get("training", {}).get("batch_size", 1024))
    nw = int(cfg.get("training", {}).get("num_workers", 4))
    pm = bool(cfg.get("training", {}).get("pin_memory", True))

    train_loader = DataLoader(TensorDataset(X_tr, z_tr), batch_size=bs, shuffle=True,  num_workers=nw, pin_memory=pm)
    val_loader   = DataLoader(TensorDataset(X_va, z_va), batch_size=bs, shuffle=False, num_workers=nw, pin_memory=pm)

    # --- Build flow
    nf_cfg = cfg.get("nf", {})
    flow = _build_flow(
        cond_dim=cond_dim,
        transform=nf_cfg.get("transform", "affine"),
        hidden_features=int(nf_cfg.get("hidden_features", 256)),
        num_layers=int(nf_cfg.get("num_layers", 6)),
        actnorm=bool(nf_cfg.get("actnorm", True)),
        num_bins=int(nf_cfg.get("num_bins", 8)),
    ).to(device)

    opt = optim.Adam(
        flow.parameters(),
        lr=float(cfg.get("training", {}).get("lr", 5e-4)),
        weight_decay=float(cfg.get("training", {}).get("weight_decay", 0.0)),
    )
    epochs = int(cfg.get("training", {}).get("epochs", 50))

    # --- CSV logging compatible with Optuna worker pruning
    csv_path = outdir / "metrics.csv"
    if not csv_path.exists():
        with open(csv_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["epoch", "split", "nll", "val_loss", "objective"])

    def _nll_epoch(loader, train: bool) -> float:
        losses = []
        if train:
            flow.train()
        else:
            flow.eval()
        with torch.set_grad_enabled(train):
            for xb, zb in loader:
                xb = xb.to(device, non_blocking=True)
                zb = zb.to(device, non_blocking=True)
                log_prob = flow.log_prob(inputs=zb, context=xb)  # log p(z | x)
                loss = -log_prob.mean()
                if train:
                    opt.zero_grad(set_to_none=True)
                    loss.backward()
                    opt.step()
                losses.append(loss.item())
        return float(np.mean(losses)) if losses else float("inf")

    best_val = float("inf")
    best_state = None

    for e in range(1, epochs + 1):
        tr = _nll_epoch(train_loader, True)
        va = _nll_epoch(val_loader, False)
        if va < best_val:
            best_val = va
            best_state = {k: v.detach().cpu() for k, v in flow.state_dict().items()}

        with open(csv_path, "a", newline="") as f:
            w = csv.writer(f)
            w.writerow([e, "train", f"{tr:.6f}", "", ""])
            w.writerow([e, "val",   f"{va:.6f}", f"{va:.6f}", f"{va:.6f}"])

        print(f"[nf-aleatoric] epoch {e}/{epochs} train_nll={tr:.4f} val_nll={va:.4f}", flush=True)

    # persist best
    if best_state is not None:
        torch.save({"flow_state_dict": best_state}, outdir / "model.pt")

    # HPO-readable metrics
    with open(outdir / "metrics.json", "w") as f:
        json.dump(
            {"objective_name": "val_nll", "objective": float(best_val), "val_loss": float(best_val)},
            f, indent=4,
        )

    # Keep the exact config used + quick run meta
    _save_yaml(cfg, outdir / "used_config.yaml")
    duration_sec = time.perf_counter() - start_time
    duration_int = int(round(duration_sec))
    h = duration_int // 3600
    m = (duration_int % 3600) // 60
    s = duration_int % 60
    duration_hms = f"{h:02d}:{m:02d}:{s:02d}"
    run_meta = {
        "role": "aleatoric",
        "head_type": _infer_head_type(cfg),
        "cond_dim": int(cond_dim),
        "nf": {
            "transform": nf_cfg.get("transform", "affine"),
            "num_layers": int(nf_cfg.get("num_layers", 6)),
            "hidden_features": int(nf_cfg.get("hidden_features", 256)),
            "actnorm": bool(nf_cfg.get("actnorm", True)),
            "num_bins": int(nf_cfg.get("num_bins", 8)),
        },
        "device": str(device),
        "seed": int(seed),
        "flow_dumps": {
            "auto_created": auto_create,
            "output_dir": str(flow_outdir),
            "base_preds": base_preds_cfg,
            "base_run": {
                "config_path": str(base_cfg_path) if base_cfg_path else "",
                "model_dir": str(base_model_dir) if base_model_dir else "",
            },
            "created": flow_created,
        },
        "timing": {
            "start_utc": start_utc,
            "end_utc": datetime.utcnow().isoformat() + "Z",
            "duration_sec": float(duration_sec),
            "duration_hms": duration_hms,
        },
    }
    with open(outdir / "run_meta.json", "w") as f:
        json.dump(run_meta, f, indent=4)

if __name__ == "__main__":
    main()
