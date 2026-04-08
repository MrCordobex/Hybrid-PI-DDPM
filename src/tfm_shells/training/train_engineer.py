from __future__ import annotations

from pathlib import Path
from typing import Any

from tfm_shells.utils.matplotlib_backend import configure_matplotlib_backend

configure_matplotlib_backend()

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from tfm_shells.config import load_config, resolve_project_path
from tfm_shells.data.dataset import ShellDataset, compute_normalization_stats
from tfm_shells.data.index import build_dataset_index, dataset_summary, filter_records, split_records
from tfm_shells.models.factory import build_scheduler, build_unet, count_parameters
from tfm_shells.training.common import (
    expand_physics_stats,
    finalize_latest_symlink,
    make_run_name,
    plot_training_curves,
    prepare_run_directories,
    resolve_device,
    save_history,
    save_run_metadata,
    seed_everything,
    timestep_weights,
)
from tfm_shells.utils.io import save_json
from tfm_shells.utils.physics import (
    branchwise_supervised_losses,
    compute_membrane_factor_from_prediction,
    compute_physical_residual,
)
from tfm_shells.utils.tracking import ExperimentTracker


def _epoch_lambda(config: dict[str, Any], epoch_index: int) -> float:
    warmup = int(config["training"]["warmup_epochs"])
    lambda_max = float(config["training"]["lambda_max"])
    total_epochs = int(config["training"]["epochs"])
    if epoch_index < warmup:
        return 0.0
    progress = (epoch_index - warmup) / max(total_epochs - warmup, 1)
    return lambda_max * min(max(progress, 0.0), 1.0)


def _run_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    scheduler,
    optimizer: torch.optim.Optimizer | None,
    stats: dict[str, Any],
    config: dict[str, Any],
    device: torch.device,
    lambda_epoch: float,
    include_fz_channel: bool,
) -> dict[str, float]:
    is_train = optimizer is not None
    model.train(is_train)
    use_amp = bool(config["training"]["mixed_precision"]) and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    autocast_device = "cuda" if device.type == "cuda" else "cpu"
    autocast_dtype = torch.float16 if device.type == "cuda" else torch.bfloat16

    total_loss = 0.0
    total_mse = 0.0
    total_phys = 0.0
    total_mf_pred = 0.0
    total_mf_true = 0.0
    total_mf_mae = 0.0
    total_uz_mse = 0.0
    total_membrane_mse = 0.0
    total_flexion_mse = 0.0
    total_items = 0

    power = float(config["training"]["timestep_power"])
    t_max = int(scheduler.config.num_train_timesteps)

    context = torch.enable_grad() if is_train else torch.no_grad()
    with context:
        for batch in tqdm(loader, leave=False):
            z_clean = batch["z"].to(device, non_blocking=True)
            fz_cond = batch["fz_norm"].to(device, non_blocking=True)
            fz_real = batch["fz_real"].to(device, non_blocking=True)
            physics_clean = batch["physics"].to(device, non_blocking=True)
            ds = batch["ds"].to(device, non_blocking=True)
            dv = batch["dv"].to(device, non_blocking=True)
            mf_true = batch["mf_true"].to(device, non_blocking=True)
            batch_size = z_clean.shape[0]

            p_mean, p_std = expand_physics_stats(stats, batch_size, device)

            noise = torch.randn_like(z_clean)
            timesteps = torch.randint(0, t_max, (batch_size,), device=device, dtype=torch.long)
            z_noisy = scheduler.add_noise(z_clean, noise, timesteps)

            if is_train:
                optimizer.zero_grad(set_to_none=True)

            with torch.autocast(device_type=autocast_device, dtype=autocast_dtype, enabled=use_amp):
                model_input = torch.cat([z_noisy, fz_cond], dim=1) if include_fz_channel else z_noisy
                pred_norm = model(model_input, timesteps).sample
                loss_mse = F.mse_loss(pred_norm, physics_clean)
                branch_losses = branchwise_supervised_losses(pred_norm, physics_clean)

                if lambda_epoch > 0.0:
                    phys_per_sample = compute_physical_residual(pred_norm, p_mean, p_std, ds, dv, fz_real)
                    weighted_phys = (phys_per_sample * timestep_weights(timesteps, t_max, power)).mean()
                else:
                    weighted_phys = torch.tensor(0.0, device=device)
                loss = loss_mse + lambda_epoch * weighted_phys

            if is_train:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(config["training"]["grad_clip_norm"]))
                scaler.step(optimizer)
                scaler.update()

            with torch.no_grad():
                model_input_t0 = torch.cat([z_clean, fz_cond], dim=1) if include_fz_channel else z_clean
                pred_t0 = model(model_input_t0, torch.zeros(batch_size, device=device, dtype=torch.long)).sample
                mf_pred_map = compute_membrane_factor_from_prediction(pred_t0, p_mean, p_std)
                mf_pred_mean = mf_pred_map.mean(dim=(1, 2, 3))
                mf_true_mean = mf_true.mean(dim=(1, 2, 3))
                mf_mae = torch.abs(mf_pred_map - mf_true).mean(dim=(1, 2, 3))

            total_loss += float(loss.item()) * batch_size
            total_mse += float(loss_mse.item()) * batch_size
            total_phys += float(weighted_phys.item()) * batch_size
            total_mf_pred += float(mf_pred_mean.mean().item()) * batch_size
            total_mf_true += float(mf_true_mean.mean().item()) * batch_size
            total_mf_mae += float(mf_mae.mean().item()) * batch_size
            total_uz_mse += float(branch_losses["uz_mse"].item()) * batch_size
            total_membrane_mse += float(branch_losses["membrane_mse"].item()) * batch_size
            total_flexion_mse += float(branch_losses["flexion_mse"].item()) * batch_size
            total_items += batch_size

    return {
        "loss": total_loss / max(total_items, 1),
        "mse": total_mse / max(total_items, 1),
        "phys": total_phys / max(total_items, 1),
        "mf_pred": total_mf_pred / max(total_items, 1),
        "mf_true": total_mf_true / max(total_items, 1),
        "mf_mae": total_mf_mae / max(total_items, 1),
        "uz_mse": total_uz_mse / max(total_items, 1),
        "membrane_mse": total_membrane_mse / max(total_items, 1),
        "flexion_mse": total_flexion_mse / max(total_items, 1),
    }


def _save_validation_figure(
    model: torch.nn.Module,
    loader: DataLoader,
    stats: dict[str, Any],
    device: torch.device,
    output_path: Path,
    include_fz_channel: bool,
) -> None:
    batch = next(iter(loader))
    z_clean = batch["z"].to(device)
    fz_cond = batch["fz_norm"].to(device)
    mf_true = batch["mf_true"].to(device)

    batch_size = z_clean.shape[0]
    p_mean, p_std = expand_physics_stats(stats, batch_size, device)

    model.eval()
    with torch.no_grad():
        model_input = torch.cat([z_clean, fz_cond], dim=1) if include_fz_channel else z_clean
        pred_t0 = model(model_input, torch.zeros(batch_size, device=device, dtype=torch.long)).sample
        mf_pred = compute_membrane_factor_from_prediction(pred_t0, p_mean, p_std)

    n_show = min(4, batch_size)
    figure, axes = plt.subplots(n_show, 3, figsize=(12, 3.2 * n_show))
    if n_show == 1:
        axes = np.array([axes])

    for row in range(n_show):
        axes[row, 0].imshow(z_clean[row, 0].cpu().numpy(), cmap="viridis")
        axes[row, 0].set_title("z_norm")
        axes[row, 0].axis("off")

        axes[row, 1].imshow(mf_true[row, 0].cpu().numpy(), cmap="magma", vmin=0.0, vmax=1.0)
        axes[row, 1].set_title("mf_true")
        axes[row, 1].axis("off")

        axes[row, 2].imshow(mf_pred[row, 0].cpu().numpy(), cmap="magma", vmin=0.0, vmax=1.0)
        axes[row, 2].set_title("mf_pred")
        axes[row, 2].axis("off")

    figure.suptitle("Engineer validation snapshots")
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def train_engineer(config_path: str | Path) -> dict[str, Any]:
    config = load_config(config_path)
    seed_everything(int(config["seed"]))
    directories = prepare_run_directories(config, role="engineer")
    device = resolve_device(str(config["runtime"]["device"]))

    dataset_dir = resolve_project_path(config, config["data"]["dataset_dir"])
    records = build_dataset_index(dataset_dir)
    filtered = filter_records(
        records,
        subset=str(config["data"]["subset"]),
        min_mf_mean=config["data"].get("min_mf_mean"),
    )
    train_records, val_records = split_records(
        filtered,
        val_ratio=float(config["data"]["val_ratio"]),
        seed=int(config["seed"]),
    )

    stats = compute_normalization_stats(train_records, include_physics=True)
    splits = {
        "train": [record["name"] for record in train_records],
        "val": [record["name"] for record in val_records],
        "dataset_summary": {
            "all_filtered": dataset_summary(filtered),
            "train": dataset_summary(train_records),
            "val": dataset_summary(val_records),
        },
    }
    save_run_metadata(config, directories, stats, splits)

    train_dataset = ShellDataset(train_records, stats=stats, include_physics=True)
    val_dataset = ShellDataset(val_records, stats=stats, include_physics=True)

    train_loader = DataLoader(
        train_dataset,
        batch_size=int(config["data"]["batch_size"]),
        shuffle=True,
        num_workers=int(config["data"]["num_workers"]),
        pin_memory=bool(config["data"]["pin_memory"]),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=int(config["data"]["batch_size"]),
        shuffle=False,
        num_workers=int(config["data"]["num_workers"]),
        pin_memory=bool(config["data"]["pin_memory"]),
    )

    model = build_unet(config["model"]).to(device)
    scheduler = build_scheduler(config["model"])
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["training"]["learning_rate"]),
        weight_decay=float(config["training"]["weight_decay"]),
    )
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=float(config["training"]["lr_factor"]),
        patience=int(config["training"]["lr_patience"]),
    )

    history_rows: list[dict[str, Any]] = []
    best_val_loss = float("inf")
    epochs_without_improvement = 0
    best_path = directories["model_root"] / "best.pt"
    last_path = directories["model_root"] / "last.pt"

    run_name = make_run_name(config, role="engineer")
    with ExperimentTracker(config, directories["project_root"], run_name) as tracker:
        tracker.log_config(config)
        tracker.log_artifact(directories["run_root"] / "config.yaml", artifact_path="run")
        tracker.log_artifact(directories["run_root"] / "stats.json", artifact_path="run")
        tracker.log_artifact(directories["run_root"] / "splits.json", artifact_path="run")
        tracker.log_metrics(
            {
                "dataset_count_filtered": float(len(filtered)),
                "dataset_count_train": float(len(train_records)),
                "dataset_count_val": float(len(val_records)),
                "mf_mean_filtered": float(splits["dataset_summary"]["all_filtered"]["mf_mean"]),
                "model_parameters": float(count_parameters(model)),
            }
        )

        for epoch in range(1, int(config["training"]["epochs"]) + 1):
            lambda_epoch = _epoch_lambda(config, epoch - 1)
            train_metrics = _run_epoch(
                model=model,
                loader=train_loader,
                scheduler=scheduler,
                optimizer=optimizer,
                stats=stats,
                config=config,
                device=device,
                lambda_epoch=lambda_epoch,
                include_fz_channel=bool(config["data"]["include_fz_channel"]),
            )
            val_metrics = _run_epoch(
                model=model,
                loader=val_loader,
                scheduler=scheduler,
                optimizer=None,
                stats=stats,
                config=config,
                device=device,
                lambda_epoch=lambda_epoch,
                include_fz_channel=bool(config["data"]["include_fz_channel"]),
            )
            lr_scheduler.step(val_metrics["loss"])
            current_lr = float(optimizer.param_groups[0]["lr"])

            row = {
                "epoch": epoch,
                "train_loss": train_metrics["loss"],
                "val_loss": val_metrics["loss"],
                "train_mse": train_metrics["mse"],
                "val_mse": val_metrics["mse"],
                "train_phys": train_metrics["phys"],
                "val_phys": val_metrics["phys"],
                "train_mf_pred": train_metrics["mf_pred"],
                "val_mf_pred": val_metrics["mf_pred"],
                "train_mf_true": train_metrics["mf_true"],
                "val_mf_true": val_metrics["mf_true"],
                "train_mf_mae": train_metrics["mf_mae"],
                "val_mf_mae": val_metrics["mf_mae"],
                "train_uz_mse": train_metrics["uz_mse"],
                "val_uz_mse": val_metrics["uz_mse"],
                "train_membrane_mse": train_metrics["membrane_mse"],
                "val_membrane_mse": val_metrics["membrane_mse"],
                "train_flexion_mse": train_metrics["flexion_mse"],
                "val_flexion_mse": val_metrics["flexion_mse"],
                "lambda_epoch": lambda_epoch,
                "learning_rate": current_lr,
            }
            history_rows.append(row)
            tracker.log_metrics(row, step=epoch)

            checkpoint = {
                "epoch": epoch,
                "role": "engineer",
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "model_config": config["model"],
                "scheduler_config": dict(scheduler.config),
                "normalization_stats": stats,
                "config_path": str(config["_meta"]["config_path"]),
                "dataset_summary": splits["dataset_summary"],
            }
            torch.save(checkpoint, last_path)

            if val_metrics["loss"] < best_val_loss:
                best_val_loss = val_metrics["loss"]
                epochs_without_improvement = 0
                torch.save(checkpoint, best_path)
                tracker.log_metrics({"best_val_loss": best_val_loss}, step=epoch)
            else:
                epochs_without_improvement += 1

            if epochs_without_improvement >= int(config["training"]["early_stopping_patience"]):
                break

        save_history(history_rows, directories)
        curves_path = directories["run_root"] / "engineer_curves.png"
        plot_training_curves(
            history_rows=history_rows,
            metrics=[
                ("train_loss", "val_loss"),
                ("train_mse", "val_mse"),
                ("train_mf_mae", "val_mf_mae"),
            ],
            title="Engineer training curves",
            output_path=curves_path,
        )
        val_fig_path = directories["run_root"] / "engineer_validation.png"
        _save_validation_figure(
            model,
            val_loader,
            stats,
            device,
            val_fig_path,
            include_fz_channel=bool(config["data"]["include_fz_channel"]),
        )

        tracker.log_artifact(curves_path, artifact_path="diagnostics")
        tracker.log_artifact(val_fig_path, artifact_path="diagnostics")
        tracker.log_artifact(directories["model_root"] / "history.csv", artifact_path="run")
        tracker.log_artifact(best_path, artifact_path="checkpoints")
        tracker.log_artifact(last_path, artifact_path="checkpoints")

        summary = {
            "best_val_loss": best_val_loss,
            "epochs_completed": len(history_rows),
            "best_checkpoint": str(best_path),
            "last_checkpoint": str(last_path),
        }
        save_json(summary, directories["run_root"] / "summary.json")
        save_json(summary, directories["model_root"] / "summary.json")
        tracker.log_artifact(directories["run_root"] / "summary.json", artifact_path="run")

    finalize_latest_symlink(directories["latest_root"], directories["model_root"])
    return {
        "best_val_loss": best_val_loss,
        "best_checkpoint": str(best_path),
        "last_checkpoint": str(last_path),
    }
