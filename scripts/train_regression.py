# uncertainty_quantification/scripts/train_regression.py
from __future__ import annotations
import argparse, csv, json, os, socket, time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, Tuple

import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import yaml

from data import get_dataloaders, inverse_target
from model_base import MLPRegressor, mse_loss, mae_loss, huber_loss, gaussian_nll, laplace_nll

def set_seed(seed: int):
    import random
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False

def _load_cfg(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        return yaml.safe_load(f)

def _objective_name(cfg: Dict[str, Any]) -> str:
    # default objective by head_type
    head = cfg["model"].get("head_type", "point").lower()
    if head == "point":
        return cfg.get("objective", "val_mae")  # Decision 3: val_mae when not probabilistic
    return cfg.get("objective", "val_nll")

def _point_loss(kind: str, mu, y):
    kind = kind.lower()
    if kind == "mae":   return mae_loss(mu, y)
    if kind == "mse":   return mse_loss(mu, y)
    if kind == "huber": return huber_loss(mu, y, delta=1.0)
    raise ValueError(f"point_loss {kind} not in [mae|mse|huber]")

def _metrics_from_batches(
    preds,
    targets,
    head_type,
    nll_values=None,
    scales=None,
) -> Dict[str, float]:
    """
    Aggregate basic metrics from batched predictions/targets.

    Notes
    -----
    - For `head_type == "point"`, this behaves exactly as before
      (MAE/RMSE only).
    - For probabilistic heads ("gauss" / "laplace"), NLL is added if
      `nll_values` is provided, and optional scale summaries are added
      if `scales` is provided.
    """
    mu = np.concatenate(preds, axis=0).reshape(-1)
    yt = np.concatenate(targets, axis=0).reshape(-1)
    ae = np.abs(mu - yt)
    se = (mu - yt) ** 2
    out = {
        "mae": float(np.mean(ae)),
        "rmse": float(np.sqrt(np.mean(se))),
    }
    if head_type in ("gauss", "laplace"):
        if nll_values is not None and len(nll_values):
            out["nll"] = float(np.mean(nll_values))
        if scales is not None and len(scales):
            s = np.concatenate(scales, axis=0).reshape(-1)
            out["scale_mean"] = float(np.mean(s))
            out["scale_median"] = float(np.median(s))
    return out


def _require_keys(d: Dict[str, Any], keys, prefix: str = ""):
    """
    Raise a clear ValueError if any of the required keys are missing in dict d.
    """
    missing = [k for k in keys if k not in d]
    if missing:
        pref = f"{prefix}." if prefix else ""
        raise ValueError(f"Missing required config key(s): {', '.join(pref + k for k in missing)}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--outdir", required=True)
    args = p.parse_args()

    cfg = _load_cfg(args.config)
    Path(args.outdir).mkdir(parents=True, exist_ok=True)
    # --- Config validation: fail fast on missing required keys ---
    _require_keys(cfg, ["data", "model", "training", "objective", "io"])

    _require_keys(cfg["data"], ["csv_path", "target_col", "split_fracs"], prefix="data")
    _require_keys(cfg["training"], ["lr", "epochs", "device", "point_loss"], prefix="training")
    _require_keys(cfg["model"], ["head_type", "hidden_dims", "activation"], prefix="model")
    # io.outdir and seed already have reasonable defaults, so we can leave them as optional.
    # --- End config validation ---


    # Make sure data.py writes preproc_meta.json into the same outdir
    # that we pass on the command line.
    cfg.setdefault("io", {})
    cfg["io"]["outdir"] = args.outdir


    seed = int(cfg.get("seed", 42)); set_seed(seed)
    device = torch.device("cuda" if (torch.cuda.is_available() and str(cfg.get("training", {}).get("device","cuda")).lower()=="cuda") else "cpu")

    # persist effective config
    with open(Path(args.outdir) / "used_config.yaml", "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)

    # data
    train_loader, val_loader, test_loader, preproc_meta = get_dataloaders(cfg, seed=seed)

    # model
    in_dim = int(preproc_meta["feature_dim"])
    hidden = cfg["model"].get("hidden_dims", [512, 256, 128])
    head_type = cfg["model"].get("head_type", "point").lower()
    model = MLPRegressor(
        in_dim=in_dim,
        hidden_dims=hidden,
        head_type=head_type,
        activation=cfg["model"].get("activation", "relu"),
        dropout=float(cfg["model"].get("dropout", 0.1)),
        use_batchnorm=bool(cfg["model"].get("batchnorm", True)),
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[model] in_dim={in_dim} hidden={hidden} head={head_type} params={n_params:,}")

    # optimizer
    train_cfg = cfg["training"]
    lr = float(train_cfg["lr"])
    wd = float(train_cfg.get("weight_decay", 0.0))
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    grad_clip = float(train_cfg.get("grad_clip", 0.0))

    epochs = max(1, int(train_cfg.get("epochs", 10)))
    eval_after_train = bool(train_cfg.get("eval_after_train", False))

    objective_name = _objective_name(cfg)

    if head_type == "point" and objective_name == "val_nll":
        raise ValueError("objective=val_nll requires a probabilistic head (gauss|laplace).")  # NEW


    # --- Early stopping config (optional) ---
    es_cfg = train_cfg.get("early_stopping", {}) or {}

    es_enabled = bool(es_cfg.get("enabled", False))
    es_patience = int(es_cfg.get("patience", 10))
    es_min_delta = float(es_cfg.get("min_delta", 0.0))
    es_counter = 0  # epochs since last improvement

    # CSV logging
    metrics_csv = Path(args.outdir) / "metrics.csv"
    write_header = not metrics_csv.exists()
    best_val = float("inf")
    best_epoch = -1
    best_state = None
    best_val_loss_at_best_epoch = None
    best_val_metrics = None  # NEW: keep val metrics at best epoch

    print(f"[train] host={socket.gethostname()} device={device} head={head_type} epochs={epochs}")

    wall_t0 = time.time()
    epoch_times = []

    for epoch in range(1, epochs + 1):
        ep_t0 = time.time()
        should_stop = False  # reset flag each epoch
        # ---- train ----
        model.train()
        train_losses = []
        train_preds, train_targets = [], []
        train_nll_vals = []
        train_scales = []  # NEW: per-batch sigma/b for gauss/laplace

        for xb, yb in train_loader:
            xb = xb.to(device, non_blocking=True); yb = yb.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            out = model(xb)
            if head_type == "point":
                mu = out["mu"]
                loss = _point_loss(cfg["training"].get("point_loss","huber"), mu, yb)
            elif head_type == "gauss":
                mu, sigma = out["mu"], out["sigma"]
                loss = gaussian_nll(mu, sigma, yb)
                train_nll_vals.append(loss.detach().item())
                train_scales.append(sigma.detach().cpu().numpy())  # NEW
            elif head_type == "laplace":
                mu, b = out["mu"], out["b"]
                loss = laplace_nll(mu, b, yb)
                train_nll_vals.append(loss.detach().item())
                train_scales.append(b.detach().cpu().numpy())      # NEW
            else:
                raise ValueError(f"Unsupported head_type={head_type}")


            loss.backward()
            if grad_clip and grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

            train_losses.append(loss.item())
            train_preds.append(out["mu"].detach().cpu().numpy())
            train_targets.append(yb.detach().cpu().numpy())

        train_metrics = _metrics_from_batches(
            train_preds,
            train_targets,
            head_type,
            nll_values=train_nll_vals,
            scales=train_scales if train_scales else None,
        )

        train_loss_scalar = np.mean(train_losses).item() if len(train_losses) else 0.0

        # ---- validate ----
        model.eval()
        val_losses = []
        val_preds, val_targets = [], []
        val_nll_vals = []
        val_scales = []  # NEW: per-batch sigma/b for gauss/laplace

        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device, non_blocking=True); yb = yb.to(device, non_blocking=True)
                out = model(xb)
                if head_type == "point":
                    mu = out["mu"]
                    vloss = _point_loss(cfg["training"].get("point_loss","huber"), mu, yb)
                elif head_type == "gauss":
                    mu, sigma = out["mu"], out["sigma"]
                    vloss = gaussian_nll(mu, sigma, yb)
                    val_nll_vals.append(vloss.item())
                    val_scales.append(sigma.detach().cpu().numpy())  # NEW
                else:
                    mu, b = out["mu"], out["b"]
                    vloss = laplace_nll(mu, b, yb)
                    val_nll_vals.append(vloss.item())
                    val_scales.append(b.detach().cpu().numpy())      # NEW


                val_losses.append(vloss.item())
                val_preds.append(out["mu"].detach().cpu().numpy())
                val_targets.append(yb.detach().cpu().numpy())

        if len(val_preds):
            # Normal case: we have validation batches, compute metrics from them
            val_metrics = _metrics_from_batches(
                val_preds,
                val_targets,
                head_type,
                nll_values=val_nll_vals,
                scales=val_scales if val_scales else None,  # NEW
            )
            val_loss_scalar = float(np.mean(val_losses)) if len(val_losses) else float("inf")
        else:
            # Edge case: no validation data (e.g., split_fracs left 0 rows for val).
            # Mirror training metrics so logging and objective wiring still work.
            val_metrics = {
                "mae": float(train_metrics["mae"]),
                "rmse": float(train_metrics["rmse"]),
            }
            if head_type in ("gauss", "laplace"):
                if len(train_nll_vals):
                    # Reuse training NLL values for logging if relevant
                    val_nll_vals = train_nll_vals.copy()
                # Also mirror scale summaries if available
                if "scale_mean" in train_metrics:
                    val_metrics["scale_mean"] = float(train_metrics["scale_mean"])
                if "scale_median" in train_metrics:
                    val_metrics["scale_median"] = float(train_metrics["scale_median"])
            # For loss, just reuse the scalar training loss
            val_loss_scalar = float(train_loss_scalar)

        # Original-space metrics (for interpretability only; do NOT affect objective)
        target_meta = preproc_meta.get("target", {})
        try:
            # Training split
            train_mu_t = np.concatenate(train_preds, axis=0).reshape(-1, 1)
            train_yt_t = np.concatenate(train_targets, axis=0).reshape(-1, 1)
            train_mu_o = inverse_target(train_mu_t, target_meta).reshape(-1)
            train_yt_o = inverse_target(train_yt_t, target_meta).reshape(-1)
            ae_o_tr = np.abs(train_mu_o - train_yt_o)
            se_o_tr = (train_mu_o - train_yt_o) ** 2
            train_metrics["mae_orig"] = float(np.mean(ae_o_tr))
            train_metrics["rmse_orig"] = float(np.sqrt(np.mean(se_o_tr)))

            # Validation split
            if len(val_preds):
                val_mu_t = np.concatenate(val_preds, axis=0).reshape(-1, 1)
                val_yt_t = np.concatenate(val_targets, axis=0).reshape(-1, 1)
                val_mu_o = inverse_target(val_mu_t, target_meta).reshape(-1)
                val_yt_o = inverse_target(val_yt_t, target_meta).reshape(-1)
                ae_o_val = np.abs(val_mu_o - val_yt_o)
                se_o_val = (val_mu_o - val_yt_o) ** 2
                val_metrics["mae_orig"] = float(np.mean(ae_o_val))
                val_metrics["rmse_orig"] = float(np.sqrt(np.mean(se_o_val)))
            else:
                # No val split: mirror training original-space metrics
                val_metrics["mae_orig"] = float(train_metrics.get("mae_orig", float("nan")))
                val_metrics["rmse_orig"] = float(train_metrics.get("rmse_orig", float("nan")))
        except Exception:
            # If inverse transform fails for any reason, keep log-space metrics only
            train_metrics.setdefault("mae_orig", float("nan"))
            train_metrics.setdefault("rmse_orig", float("nan"))
            val_metrics.setdefault("mae_orig", float("nan"))
            val_metrics.setdefault("rmse_orig", float("nan"))


        # objective wiring: scalar objective_val for this epoch
        if objective_name == "val_mae":
            objective_val = float(val_metrics["mae"])
        elif objective_name == "val_rmse":
            objective_val = float(val_metrics["rmse"])
        elif objective_name == "val_mae_orig":           # NEW: target-space MAE
            objective_val = float(val_metrics["mae_orig"])
        elif objective_name == "val_rmse_orig":          # NEW: target-space RMSE
            objective_val = float(val_metrics["rmse_orig"])
        else:  # "val_nll"
            # prefer model-appropriate NLL if available, else fall back to the scalar vloss
            objective_val = float(np.mean(val_nll_vals)) if len(val_nll_vals) else val_loss_scalar



        # best tracking + early stopping
        improved = False
        if es_enabled:
            # Treat as improvement only if it beats best_val by at least min_delta
            if objective_val < best_val - es_min_delta:
                best_val, best_epoch = objective_val, epoch
                es_counter = 0
                should_stop = False
                improved = True
            else:
                es_counter += 1
                should_stop = es_counter >= es_patience
        else:
            if objective_val < best_val:
                best_val, best_epoch = objective_val, epoch
                improved = True

        # snapshot best-epoch weights so we can save them after training
        if improved:
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            best_val_loss_at_best_epoch = val_loss_scalar
            # NEW: snapshot validation metrics at best epoch (for prob. heads)
            best_val_metrics = dict(val_metrics)


        # durations
        ep_dur = round(time.time() - ep_t0, 3)
        epoch_times.append(ep_dur)

        # CSV append
        ts = datetime.now(timezone.utc).isoformat(timespec="seconds")
        with open(metrics_csv, "a", newline="") as mf:
            w = csv.writer(mf)
            if write_header:
                w.writerow([
                    "epoch",
                    "split",
                    "loss",
                    "mae",
                    "rmse",
                    "mae_orig",
                    "rmse_orig",
                    "nll",
                    "elapsed_sec",
                    "timestamp",
                    "val_loss",
                    "objective",
                ])
                write_header = False
            # train (log-space + original-space metrics)
            w.writerow([
                epoch,
                "train",
                f"{train_loss_scalar:.4f}",
                f"{train_metrics['mae']:.4f}",
                f"{train_metrics['rmse']:.4f}",
                f"{train_metrics.get('mae_orig', float('nan')):.4f}",
                f"{train_metrics.get('rmse_orig', float('nan')):.4f}",
                f"{np.mean(train_nll_vals) if train_nll_vals else 0.0:.4f}",
                f"{ep_dur:.3f}",
                ts,
                "",
                "",
            ])
            # val: fill val_loss (for pruner) and objective column
            w.writerow([
                epoch,
                "val",
                f"{val_loss_scalar:.4f}",
                f"{val_metrics['mae']:.4f}",
                f"{val_metrics['rmse']:.4f}",
                f"{val_metrics.get('mae_orig', float('nan')):.4f}",
                f"{val_metrics.get('rmse_orig', float('nan')):.4f}",
                f"{np.mean(val_nll_vals) if val_nll_vals else 0.0:.4f}",
                "",
                ts,
                f"{val_loss_scalar:.4f}",
                f"{objective_val:.4f}",
            ])


        msg = (
            f"[epoch {epoch}] \t"
            f"train_loss={train_loss_scalar:.4f}\t "
            f"val_loss={val_loss_scalar:.4f}\t "
            f"val_mae_orig={val_metrics.get('mae_orig', float('nan')):.4f}\t "
            f"OBJ: {objective_name}={objective_val:.4f}\t"
        )
        if head_type in ("gauss", "laplace"):
            msg += (
                f" mean_scale={val_metrics.get('scale_mean', float('nan')):.4f}\t"
                f" median_scale={val_metrics.get('scale_median', float('nan')):.4f}"
            )
        print(msg)


        if es_enabled and should_stop:
            print(
                f"[early_stop] Stopping after epoch {epoch}: "
                f"no improvement in {es_patience} epochs "
                f"(best {objective_name}={best_val:.4f} at epoch {best_epoch})"
            )
            break

    # save model
    # save model — ensure we save the BEST epoch, not the last
    if best_state is not None:
        model.load_state_dict(best_state)
    torch.save({"model_state_dict": model.state_dict()}, Path(args.outdir) / "model.pt")


    # run meta + metrics.json (HPO reads this)
    end_ts = datetime.now(timezone.utc).isoformat(timespec="seconds")

    # Aggregate training time
    total_train_sec = float(sum(epoch_times))
    total_train_sec_rounded = round(total_train_sec, 3)
    total_train_int = int(round(total_train_sec))
    hours = total_train_int // 3600
    minutes = (total_train_int % 3600) // 60
    seconds = total_train_int % 60
    train_time_hms = f"{hours:02d}:{minutes:02d}:{seconds:02d}"

    run_meta = {
        "host": socket.gethostname(),
        "device": str(device),
        "torch": torch.__version__,
        "epochs": int(epochs),
        "seed": int(seed),
        # Approximate start time from total_train_sec (keeps previous semantics)
        "start_utc": datetime.fromtimestamp(time.time() - total_train_sec, tz=timezone.utc).isoformat(timespec="seconds"),
        "end_utc": end_ts,
        "train_time_sec": total_train_sec_rounded,
        "train_time_hms": train_time_hms,
        "epoch_times_sec": epoch_times,
        "best_epoch": best_epoch,
    }
    with open(Path(args.outdir) / "run_meta.json", "w") as f:
        json.dump(run_meta, f, indent=4)

    # --- HPO payload: metrics.json expected by the Optuna worker ---
    # Use the best value of the chosen objective over epochs.
    # best_val tracks the min objective_val; best_val_loss_at_best_epoch is the val_loss there.
    metrics_payload = {
        "objective_name": objective_name,
        "objective": float(best_val if best_val not in (None, float("inf")) else objective_val),
        "best_epoch": int(best_epoch),
        "val_loss_at_best_epoch": (
            float(best_val_loss_at_best_epoch)
            if best_val_loss_at_best_epoch is not None
            else float("nan")
        ),
        # Optional extras (handy when debugging HPO)
        "train_time_sec": total_train_sec_rounded,
        "train_time_hms": train_time_hms,
    }

    # For probabilistic heads, also expose validation-scale summaries at the best epoch
    if head_type in ("gauss", "laplace") and best_val_metrics is not None:
        if "scale_mean" in best_val_metrics:
            metrics_payload["val_scale_mean_at_best"] = float(best_val_metrics["scale_mean"])
        if "scale_median" in best_val_metrics:
            metrics_payload["val_scale_median_at_best"] = float(best_val_metrics["scale_median"])
        # NEW: include MAE/RMSE in target space at best epoch
        if "mae_orig" in best_val_metrics:
            metrics_payload["val_mae_orig_at_best"] = float(best_val_metrics["mae_orig"])
        if "rmse_orig" in best_val_metrics:
            metrics_payload["val_rmse_orig_at_best"] = float(best_val_metrics["rmse_orig"])

    with open(Path(args.outdir) / "metrics.json", "w") as f:
        json.dump(metrics_payload, f, indent=4)


    if eval_after_train:
        # Where to put eval results; configurable via io.evals_root, default "outputs/evals"
        evals_root = cfg.get("io", {}).get("evals_root", "outputs/evals")
        evals_root = Path(evals_root)
        run_tag = Path(args.outdir).name
        eval_dir = evals_root / run_tag
        eval_dir.mkdir(parents=True, exist_ok=True)

        # ---- Save TRAIN and VAL predictions (original target scale) ----
        model.eval()

        for split_name, loader, split_key, csv_name in [
            ("train", train_loader, "train", "train_preds.csv"),
            ("val",   val_loader,   "val",   "val_preds.csv"),
        ]:
            if loader is None:
                continue  # defensive: if a split is missing, skip

            split_losses = []
            split_preds, split_targets = [], []
            split_nll_vals = []
            split_scales = []  # NEW: per-batch sigma/b for this split

            with torch.no_grad():
                for xb, yb in loader:
                    xb = xb.to(device, non_blocking=True)
                    yb = yb.to(device, non_blocking=True)
                    out = model(xb)

                if head_type == "point":
                    mu = out["mu"]
                    tloss = _point_loss(cfg["training"].get("point_loss", "huber"), mu, yb)
                elif head_type == "gauss":
                    mu, sigma = out["mu"], out["sigma"]
                    tloss = gaussian_nll(mu, sigma, yb)
                    split_scales.append(sigma.detach().cpu().numpy())  # NEW
                elif head_type == "laplace":
                    mu, b = out["mu"], out["b"]
                    tloss = laplace_nll(mu, b, yb)
                    split_scales.append(b.detach().cpu().numpy())      # NEW
                else:
                    raise ValueError(f"Unknown head_type {head_type}")

                split_losses.append(tloss.detach().cpu().numpy())
                split_preds.append(mu.detach().cpu().numpy())
                split_targets.append(yb.detach().cpu().numpy())
                if head_type in ("gauss", "laplace"):
                    split_nll_vals.append(tloss.detach().cpu().numpy())

            if not split_preds:
                # No data for this split – nothing to save
                continue

            # Metrics on transformed target scale (not strictly needed just to save preds)
            # Metrics on transformed target scale (not strictly needed just to save preds)
            split_metrics = _metrics_from_batches(
                split_preds,
                split_targets,
                head_type,
                nll_values=split_nll_vals,
                scales=split_scales if split_scales else None,  # NEW
            )

            split_loss_scalar = float(np.mean(split_losses)) if len(split_losses) else float("inf")
            split_metrics["loss"] = split_loss_scalar

            # Metrics on original target scale
            mu_t = np.concatenate(split_preds, axis=0).reshape(-1, 1)
            yt_t = np.concatenate(split_targets, axis=0).reshape(-1, 1)

            target_meta = preproc_meta.get("target", {})
            mu_o = inverse_target(mu_t, target_meta).reshape(-1)
            yt_o = inverse_target(yt_t, target_meta).reshape(-1)
            split_scales_all = None
            if head_type in ("gauss", "laplace") and split_scales:
                split_scales_all = np.concatenate(split_scales, axis=0).reshape(-1)

            # Align with original IDs or row indices for this split
            splits = preproc_meta.get("splits", {})
            split_idx = np.array(splits.get(split_key, []), dtype=int)

            id_col = preproc_meta.get("id_col")
            ids_all = preproc_meta.get("id_values")
            if id_col is not None and ids_all is not None:
                ids_all = np.array(ids_all)
                ids_split = ids_all[split_idx]
                first_col = ids_split
                id_header = id_col
            else:
                first_col = split_idx
                id_header = "row_index"

            if split_scales_all is not None:
                header = [id_header, "y_true", "y_pred", "y_scale"]
            else:
                header = [id_header, "y_true", "y_pred"]

            # Save per-instance predictions (original target scale)
            out_path = eval_dir / csv_name
            with open(out_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(header)
                if split_scales_all is not None:
                    for key, yt_i, mu_i, s_i in zip(first_col, yt_o, mu_o, split_scales_all):
                        writer.writerow([key, float(yt_i), float(mu_i), float(s_i)])
                else:
                    for key, yt_i, mu_i in zip(first_col, yt_o, mu_o):
                        writer.writerow([key, float(yt_i), float(mu_i)])


            print(f"[eval-after-train] Saved {split_name} predictions to: {out_path}")


        te_losses = []
        te_preds, te_targets = [], []
        te_nll_vals = []
        te_scales = []  # NEW: per-batch sigma/b for test split


        with torch.no_grad():
            for xb, yb in test_loader:
                xb = xb.to(device, non_blocking=True)
                yb = yb.to(device, non_blocking=True)
                out = model(xb)

                if head_type == "point":
                    mu = out["mu"]
                    tloss = _point_loss(cfg["training"].get("point_loss", "huber"), mu, yb)
                elif head_type == "gauss":
                    mu, sigma = out["mu"], out["sigma"]
                    tloss = gaussian_nll(mu, sigma, yb)
                    te_scales.append(sigma.detach().cpu().numpy())  # NEW
                elif head_type == "laplace":
                    mu, b = out["mu"], out["b"]
                    tloss = laplace_nll(mu, b, yb)
                    te_scales.append(b.detach().cpu().numpy())      # NEW
                else:
                    raise ValueError(f"Unknown head_type {head_type}")


                te_losses.append(tloss.detach().cpu().numpy())
                te_preds.append(mu.detach().cpu().numpy())
                te_targets.append(yb.detach().cpu().numpy())
                if head_type in ("gauss", "laplace"):
                    te_nll_vals.append(tloss.detach().cpu().numpy())

        # Metrics on transformed target scale
        # Metrics on transformed target scale
        test_metrics = _metrics_from_batches(
            te_preds,
            te_targets,
            head_type,
            nll_values=te_nll_vals,
            scales=te_scales if te_scales else None,  # NEW
        )

        test_loss_scalar = float(np.mean(te_losses)) if len(te_losses) else float("inf")
        test_metrics["loss"] = test_loss_scalar

        # Metrics on original target scale
        mu_t = np.concatenate(te_preds, axis=0).reshape(-1, 1)
        yt_t = np.concatenate(te_targets, axis=0).reshape(-1, 1)

        # IMPORTANT: use target meta, not whole preproc_meta
        target_meta = preproc_meta.get("target", {})
        mu_o = inverse_target(mu_t, target_meta).reshape(-1)
        yt_o = inverse_target(yt_t, target_meta).reshape(-1)

        # NEW: flatten per-instance scales (still in target-transform space)
        te_scales_all = None
        if head_type in ("gauss", "laplace") and te_scales:
            te_scales_all = np.concatenate(te_scales, axis=0).reshape(-1)


        ae_o = np.abs(mu_o - yt_o)
        se_o = (mu_o - yt_o) ** 2
        test_metrics["mae_orig"] = float(np.mean(ae_o))
        test_metrics["rmse_orig"] = float(np.sqrt(np.mean(se_o)))

        # Save test metrics
        test_payload = {
            "head_type": head_type,
            "objective_name": objective_name,
            "best_epoch": int(best_epoch),
            "test": test_metrics,
        }
        with open(eval_dir / "test_metrics.json", "w") as f:
            json.dump(test_payload, f, indent=4)

        # Build ID/row_index for the TEST split
        splits = preproc_meta.get("splits", {})
        test_idx = np.array(splits.get("test", []), dtype=int)

        id_col = preproc_meta.get("id_col")
        ids_all = preproc_meta.get("id_values")
        if id_col is not None and ids_all is not None:
            ids_all = np.array(ids_all)
            ids_test = ids_all[test_idx]
            first_col = ids_test
            id_header = id_col
        else:
            # Fallback: use row_index if no explicit ID column
            first_col = test_idx
            id_header = "row_index"

        if te_scales_all is not None:
            header = [id_header, "y_true", "y_pred", "y_scale"]
        else:
            header = [id_header, "y_true", "y_pred"]

        # Save per-instance predictions (original target scale) with ID/row_index
        with open(eval_dir / "test_preds.csv", "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(header)
            if te_scales_all is not None:
                for key, yt_i, mu_i, s_i in zip(first_col, yt_o, mu_o, te_scales_all):
                    writer.writerow([key, float(yt_i), float(mu_i), float(s_i)])
            else:
                for key, yt_i, mu_i in zip(first_col, yt_o, mu_o):
                    writer.writerow([key, float(yt_i), float(mu_i)])


        print(f"[eval-after-train] Saved test predictions to: {eval_dir / 'test_preds.csv'}")


if __name__ == "__main__":
    main()
