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
from tfm_shells.data.index import build_dataset_index, dataset_summary, filter_records
from tfm_shells.models.factory import build_scheduler, build_unet, count_parameters
from tfm_shells.training.common import (
    architect_target,
    finalize_latest_symlink,
    format_metric,
    make_run_name,
    plot_training_curves,
    prepare_run_directories,
    resolve_device,
    save_history,
    save_run_metadata,
    seed_everything,
)
from tfm_shells.utils.io import save_json
from tfm_shells.utils.tracking import ExperimentTracker


def _sample_images(
    model: torch.nn.Module,
    scheduler,
    batch_size: int,
    height: int,
    width: int,
    device: torch.device,
    num_steps: int,
) -> torch.Tensor:
    model.eval()
    inference_scheduler = build_scheduler(
        {
            "beta_schedule": scheduler.config.beta_schedule,
            "prediction_type": scheduler.config.prediction_type,
        }
    )
    inference_scheduler.set_timesteps(num_steps, device=device)

    samples = torch.randn((batch_size, 1, height, width), device=device)
    with torch.no_grad():
        for timestep in inference_scheduler.timesteps:
            t_batch = torch.full((samples.shape[0],), int(timestep.item()), device=device, dtype=torch.long)
            model_output = model(samples, t_batch).sample
            samples = inference_scheduler.step(model_output, timestep, samples).prev_sample
    return samples


def _save_sample_grid(
    samples: torch.Tensor,
    stats: dict[str, Any],
    output_path: Path,
) -> None:
    z_min = float(stats["z_min"])
    z_max = float(stats["z_max"])
    denorm = ((samples.detach().cpu().numpy() + 1.0) / 2.0) * (z_max - z_min) + z_min

    batch_size = denorm.shape[0]
    cols = min(3, batch_size)
    rows = int(np.ceil(batch_size / cols))
    figure, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3.5 * rows))
    axes = np.array(axes).reshape(-1)
    for index, axis in enumerate(axes):
        axis.axis("off")
        if index < batch_size:
            axis.imshow(denorm[index, 0], cmap="viridis")
            axis.set_title(f"sample_{index}")
    figure.suptitle("Architect conditional samples")
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def _run_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    scheduler,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    mixed_precision: bool,
    grad_clip_norm: float,
    epoch: int,
    total_epochs: int,
    phase: str,
) -> dict[str, float]:
    is_train = optimizer is not None
    model.train(is_train)
    use_amp = mixed_precision and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    autocast_device = "cuda" if device.type == "cuda" else "cpu"
    autocast_dtype = torch.float16 if device.type == "cuda" else torch.bfloat16

    total_loss = 0.0
    total_items = 0

    context = torch.enable_grad() if is_train else torch.no_grad()
    with context:
        progress = tqdm(loader, leave=False, desc=f"{phase} {epoch:03d}/{total_epochs:03d}")
        for batch in progress:
            z_clean = batch["z"].to(device, non_blocking=True)
            batch_size = z_clean.shape[0]

            noise = torch.randn_like(z_clean)
            timesteps = torch.randint(
                0,
                scheduler.config.num_train_timesteps,
                (batch_size,),
                device=device,
                dtype=torch.long,
            )
            z_noisy = scheduler.add_noise(z_clean, noise, timesteps)
            target = architect_target(scheduler, z_clean, noise, timesteps)

            if is_train:
                optimizer.zero_grad(set_to_none=True)

            with torch.autocast(device_type=autocast_device, dtype=autocast_dtype, enabled=use_amp):
                prediction = model(z_noisy, timesteps).sample
                loss = F.mse_loss(prediction, target)

            if is_train:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
                scaler.step(optimizer)
                scaler.update()

            total_loss += float(loss.item()) * batch_size
            total_items += batch_size
            progress.set_postfix(loss=format_metric(total_loss / max(total_items, 1)), refresh=False)

    return {"loss": total_loss / max(total_items, 1)}


def train_architect(config_path: str | Path) -> dict[str, Any]:
    config = load_config(config_path)
    seed_everything(int(config["seed"]))
    directories = prepare_run_directories(config, role="architect")
    device = resolve_device(str(config["runtime"]["device"]))

    dataset_dir = resolve_project_path(config, config["data"]["dataset_dir"])
    records = build_dataset_index(dataset_dir)
    filtered = filter_records(
        records,
        subset=str(config["data"]["subset"]),
        min_mf_mean=config["data"].get("min_mf_mean"),
    )

    stats = compute_normalization_stats(filtered, include_physics=False)
    splits = {
        "train": [record["name"] for record in filtered],
        "val": [],
        "dataset_summary": {
            "all_filtered": dataset_summary(filtered),
            "train": dataset_summary(filtered),
        },
    }
    save_run_metadata(config, directories, stats, splits)

    train_dataset = ShellDataset(filtered, stats=stats, include_physics=False)

    train_loader = DataLoader(
        train_dataset,
        batch_size=int(config["data"]["batch_size"]),
        shuffle=True,
        num_workers=int(config["data"]["num_workers"]),
        pin_memory=bool(config["data"]["pin_memory"]),
    )

    model = build_unet(config["model"]).to(device)
    scheduler = build_scheduler(config["model"])
    model_parameters = count_parameters(model)

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
    best_train_loss = float("inf")
    best_path = directories["model_root"] / "best.pt"
    last_path = directories["model_root"] / "last.pt"
    total_epochs = int(config["training"]["epochs"])

    run_name = make_run_name(config, role="architect")
    with ExperimentTracker(config, directories["project_root"], run_name) as tracker:
        summary_filtered = splits["dataset_summary"]["all_filtered"]
        print("\n" + "=" * 96)
        print("ARCHITECT TRAINING")
        print(f"config: {config['_meta']['config_path']}")
        print(f"device: {device}")
        print(f"dataset: {dataset_dir}")
        print(
            f"records: filtered={summary_filtered['count']} "
            f"train={len(filtered)} "
            f"subsets={summary_filtered['subset_counts']}"
        )
        print(
            f"normalization: z=[{format_metric(float(stats['z_min']))}, {format_metric(float(stats['z_max']))}] "
        )
        print(f"model_parameters: {model_parameters:,}")
        print(f"artifacts: {directories['run_root']}")
        print(f"checkpoints: {directories['model_root']}")
        print(f"mlflow_run_id: {tracker.run_id}")
        print("=" * 96)

        tracker.log_config(config)
        tracker.log_artifact(directories["run_root"] / "config.yaml", artifact_path="run")
        tracker.log_artifact(directories["run_root"] / "stats.json", artifact_path="run")
        tracker.log_artifact(directories["run_root"] / "splits.json", artifact_path="run")
        tracker.log_metrics(
            {
                "dataset_count_filtered": float(len(filtered)),
                "dataset_count_train": float(len(filtered)),
                "mf_mean_filtered": float(splits["dataset_summary"]["all_filtered"]["mf_mean"]),
                "model_parameters": float(model_parameters),
            }
        )

        for epoch in range(1, total_epochs + 1):
            train_metrics = _run_epoch(
                model=model,
                loader=train_loader,
                scheduler=scheduler,
                optimizer=optimizer,
                device=device,
                mixed_precision=bool(config["training"]["mixed_precision"]),
                grad_clip_norm=float(config["training"]["grad_clip_norm"]),
                epoch=epoch,
                total_epochs=total_epochs,
                phase="train",
            )
            lr_scheduler.step(train_metrics["loss"])
            current_lr = float(optimizer.param_groups[0]["lr"])

            row = {
                "epoch": epoch,
                "train_loss": train_metrics["loss"],
                "learning_rate": current_lr,
            }
            history_rows.append(row)
            tracker.log_metrics(row, step=epoch)

            checkpoint = {
                "epoch": epoch,
                "role": "architect",
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "model_config": config["model"],
                "scheduler_config": dict(scheduler.config),
                "normalization_stats": stats,
                "config_path": str(config["_meta"]["config_path"]),
                "dataset_summary": splits["dataset_summary"],
            }
            torch.save(checkpoint, last_path)

            if train_metrics["loss"] < best_train_loss:
                best_train_loss = train_metrics["loss"]
                torch.save(checkpoint, best_path)
                tracker.log_metrics({"best_train_loss": best_train_loss}, step=epoch)
                best_marker = " [best]"
            else:
                best_marker = ""

            tqdm.write(
                " | ".join(
                    [
                        f"epoch {epoch:03d}/{total_epochs:03d}",
                        f"train_loss={format_metric(train_metrics['loss'])}",
                        f"best_train={format_metric(best_train_loss)}{best_marker}",
                        f"lr={format_metric(current_lr)}",
                    ]
                )
            )

            sample_every = int(config["training"]["sample_every_n_epochs"])
            if epoch == 1 or epoch % sample_every == 0:
                samples = _sample_images(
                    model=model,
                    scheduler=scheduler,
                    batch_size=int(config["training"]["sample_batch_size"]),
                    height=int(config["model"]["sample_size"]),
                    width=int(config["model"]["sample_size"]),
                    device=device,
                    num_steps=int(config["training"]["sample_inference_steps"]),
                )
                sample_path = directories["run_root"] / f"samples_epoch_{epoch:03d}.png"
                _save_sample_grid(samples, stats, sample_path)
                tracker.log_artifact(sample_path, artifact_path="samples")
                tqdm.write(f"saved_samples: {sample_path}")

        save_history(history_rows, directories)
        curves_path = directories["run_root"] / "architect_curves.png"
        plot_training_curves(
            history_rows=history_rows,
            metrics=[("train_loss", None)],
            title="Architect training curves",
            output_path=curves_path,
        )
        tracker.log_artifact(curves_path, artifact_path="diagnostics")
        tracker.log_artifact(directories["model_root"] / "history.csv", artifact_path="run")
        tracker.log_artifact(best_path, artifact_path="checkpoints")
        tracker.log_artifact(last_path, artifact_path="checkpoints")

        summary = {
            "best_train_loss": best_train_loss,
            "epochs_completed": len(history_rows),
            "run_id": tracker.run_id,
            "best_checkpoint": str(best_path),
            "last_checkpoint": str(last_path),
        }
        save_json(summary, directories["run_root"] / "summary.json")
        save_json(summary, directories["model_root"] / "summary.json")
        tracker.log_artifact(directories["run_root"] / "summary.json", artifact_path="run")
        print(
            f"architect_finished | epochs={len(history_rows)} | best_train={format_metric(best_train_loss)} "
            f"| best_ckpt={best_path}"
        )

    finalize_latest_symlink(directories["latest_root"], directories["model_root"])
    return {
        "best_train_loss": best_train_loss,
        "best_checkpoint": str(best_path),
        "last_checkpoint": str(last_path),
    }
