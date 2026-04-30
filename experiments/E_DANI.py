from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
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
from tfm_shells.data.dataset import ShellDataset
from tfm_shells.data.index import build_dataset_index, filter_records, split_records
from tfm_shells.models.factory import build_scheduler, build_unet
from tfm_shells.training.common import (
    bell_guidance_weight,
    expand_physics_stats,
    resolve_device,
    seed_everything,
)
from tfm_shells.utils.physics import compute_membrane_factor_from_prediction


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ENGINEER = PROJECT_ROOT / "models" / "engineer" / "20260416_142218_engineer" / "best.pt"
DEFAULT_CLEAN = PROJECT_ROOT / "models" / "modelo_clean" / "best.pt"
FALLBACK_CLEAN = PROJECT_ROOT / "models" / "engineer" / "modelo_clean" / "best.pt"
DEFAULT_ARCHITECT = PROJECT_ROOT / "models" / "architect" / "latest" / "best.pt"
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "engineer.yaml"


@dataclass(frozen=True)
class LoadedModel:
    name: str
    model: torch.nn.Module
    checkpoint: dict[str, Any]
    scheduler: Any
    stats: dict[str, Any]
    include_fz: bool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="E_DANI robustness and guidance experiments.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--engineer-checkpoint", type=Path, default=DEFAULT_ENGINEER)
    parser.add_argument("--clean-checkpoint", type=Path, default=DEFAULT_CLEAN)
    parser.add_argument("--architect-checkpoint", type=Path, default=DEFAULT_ARCHITECT)
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "artifacts" / "E_DANI")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--timestep-stride", type=int, default=50)
    parser.add_argument("--eval-batch-size", type=int, default=4)
    parser.add_argument("--eval-max-batches", type=int, default=None)
    parser.add_argument("--sample-steps", type=int, default=1000)
    parser.add_argument("--guidance-scale", type=float, default=10.0)
    parser.add_argument("--guidance-clip", type=float, default=5.0)
    parser.add_argument("--bell-w-max", type=float, default=1.0)
    parser.add_argument("--bell-peak", type=float, default=0.5)
    parser.add_argument("--bell-width", type=float, default=0.22)
    return parser.parse_args()


def load_checkpoint(path: Path, device: torch.device) -> dict[str, Any]:
    return torch.load(path, map_location=device, weights_only=False)


def resolve_clean_checkpoint(path: Path) -> Path:
    if path.exists():
        return path
    if FALLBACK_CLEAN.exists():
        return FALLBACK_CLEAN
    raise FileNotFoundError(f"Clean checkpoint not found: {path}")


def load_surrogate(name: str, checkpoint_path: Path, device: torch.device) -> LoadedModel:
    checkpoint = load_checkpoint(checkpoint_path, device)
    model = build_unet(checkpoint["model_config"]).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return LoadedModel(
        name=name,
        model=model,
        checkpoint=checkpoint,
        scheduler=build_scheduler(checkpoint["model_config"]),
        stats=checkpoint["normalization_stats"],
        include_fz=int(checkpoint["model_config"].get("in_channels", 1)) > 1,
    )


def minmax_denorm(x_norm: torch.Tensor, minimum: float, maximum: float) -> torch.Tensor:
    return ((x_norm + 1.0) / 2.0) * (maximum - minimum) + minimum


def minmax_norm(x_real: torch.Tensor, minimum: float, maximum: float) -> torch.Tensor:
    return 2.0 * ((x_real - minimum) / (maximum - minimum + 1e-8)) - 1.0


def convert_z_between_stats(x_eval: torch.Tensor, eval_stats: dict[str, Any], model_stats: dict[str, Any]) -> torch.Tensor:
    x_real = minmax_denorm(x_eval, float(eval_stats["z_min"]), float(eval_stats["z_max"]))
    return minmax_norm(x_real, float(model_stats["z_min"]), float(model_stats["z_max"]))


def convert_fz_between_stats(fz_real: torch.Tensor, model_stats: dict[str, Any]) -> torch.Tensor:
    return minmax_norm(fz_real, float(model_stats["fz_min"]), float(model_stats["fz_max"]))


def model_prediction(
    loaded: LoadedModel,
    x_eval: torch.Tensor,
    fz_real: torch.Tensor,
    timesteps: torch.Tensor,
    eval_stats: dict[str, Any],
) -> torch.Tensor:
    x_model = convert_z_between_stats(x_eval, eval_stats, loaded.stats)
    if loaded.include_fz:
        fz_model = convert_fz_between_stats(fz_real, loaded.stats)
        model_input = torch.cat([x_model, fz_model], dim=1)
    else:
        model_input = x_model
    return loaded.model(model_input, timesteps).sample


def mf_and_objective(
    loaded: LoadedModel,
    pred_norm: torch.Tensor,
    target: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    batch_size = pred_norm.shape[0]
    p_mean, p_std = expand_physics_stats(loaded.stats, batch_size, pred_norm.device)
    mf_map = compute_membrane_factor_from_prediction(pred_norm, p_mean, p_std)
    mf_mean = mf_map.mean(dim=(1, 2, 3))
    objective = (target - mf_mean).square().sum()
    return mf_map, objective


def prediction_error_and_grad(
    loaded: LoadedModel,
    x_eval: torch.Tensor,
    fz_real: torch.Tensor,
    physics_real_target: torch.Tensor,
    mf_true: torch.Tensor,
    timesteps_for_model: torch.Tensor,
    eval_stats: dict[str, Any],
) -> tuple[float, float, float]:
    x_req = x_eval.detach().clone().requires_grad_(True)
    pred_norm = model_prediction(loaded, x_req, fz_real, timesteps_for_model, eval_stats)
    p_mean, p_std = expand_physics_stats(loaded.stats, pred_norm.shape[0], pred_norm.device)
    pred_real = pred_norm * p_std + p_mean
    mf_pred, objective = mf_and_objective(loaded, pred_norm)
    grad = torch.autograd.grad(objective, x_req, retain_graph=False, create_graph=False)[0]

    physics_mse = F.mse_loss(pred_real, physics_real_target).item()
    mf_mae = torch.abs(mf_pred - mf_true).mean().item()
    grad_norm = grad.flatten(1).norm(dim=1).mean().item()
    return float(physics_mse), float(mf_mae), float(grad_norm)


def predict_x0_from_model_output(scheduler: Any, x_t: torch.Tensor, model_output: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
    alpha_prod_t = scheduler.alphas_cumprod[timestep].to(device=x_t.device, dtype=x_t.dtype)
    alpha_prod_t = alpha_prod_t.reshape(-1, 1, 1, 1)
    beta_prod_t = 1.0 - alpha_prod_t

    prediction_type = str(scheduler.config.prediction_type)
    if prediction_type == "epsilon":
        return (x_t - beta_prod_t.sqrt() * model_output) / alpha_prod_t.sqrt()
    if prediction_type == "sample":
        return model_output
    if prediction_type == "v_prediction":
        return alpha_prod_t.sqrt() * x_t - beta_prod_t.sqrt() * model_output
    raise ValueError(f"Unsupported prediction type: {prediction_type}")


def build_validation_loader(config: dict[str, Any], stats: dict[str, Any], batch_size: int) -> DataLoader:
    dataset_dir = resolve_project_path(config, config["data"]["dataset_dir"])
    records = build_dataset_index(dataset_dir)
    filtered = filter_records(
        records,
        subset=str(config["data"]["subset"]),
        min_mf_mean=config["data"].get("min_mf_mean"),
    )
    _, val_records = split_records(filtered, val_ratio=float(config["data"]["val_ratio"]), seed=int(config["seed"]))
    dataset = ShellDataset(val_records, stats=stats, include_physics=True)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=False)


def evaluate_timestep_curves(
    engineer: LoadedModel,
    clean: LoadedModel,
    loader: DataLoader,
    args: argparse.Namespace,
    device: torch.device,
    output_dir: Path,
) -> list[dict[str, float | int | str]]:
    t_max = int(engineer.scheduler.config.num_train_timesteps)
    timesteps = list(range(0, t_max, int(args.timestep_stride)))
    if timesteps[-1] != t_max - 1:
        timesteps.append(t_max - 1)

    rows: list[dict[str, float | int | str]] = []
    generator = torch.Generator(device=device)
    generator.manual_seed(int(args.seed))

    for t_value in tqdm(timesteps, desc="E_DANI eval timesteps"):
        totals: dict[str, dict[str, float]] = {
            "engineer_time": {"physics_mse": 0.0, "mf_mae": 0.0, "grad_norm": 0.0, "n": 0.0},
            "clean_xt": {"physics_mse": 0.0, "mf_mae": 0.0, "grad_norm": 0.0, "n": 0.0},
        }
        for batch_index, batch in enumerate(loader):
            if args.eval_max_batches is not None and batch_index >= int(args.eval_max_batches):
                break
            z_clean = batch["z"].to(device)
            fz_real = batch["fz_real"].to(device)
            physics_norm = batch["physics"].to(device)
            mf_true = batch["mf_true"].to(device)
            p_mean, p_std = expand_physics_stats(engineer.stats, z_clean.shape[0], device)
            physics_real = physics_norm * p_std + p_mean

            noise = torch.randn(z_clean.shape, generator=generator, device=device, dtype=z_clean.dtype)
            t_batch = torch.full((z_clean.shape[0],), t_value, device=device, dtype=torch.long)
            t_zero = torch.zeros_like(t_batch)
            x_t = engineer.scheduler.add_noise(z_clean, noise, t_batch)

            metrics = {
                "engineer_time": prediction_error_and_grad(
                    engineer, x_t, fz_real, physics_real, mf_true, t_batch, engineer.stats
                ),
                "clean_xt": prediction_error_and_grad(
                    clean, x_t, fz_real, physics_real, mf_true, t_zero, engineer.stats
                ),
            }
            batch_size = float(z_clean.shape[0])
            for name, (physics_mse, mf_mae, grad_norm) in metrics.items():
                totals[name]["physics_mse"] += physics_mse * batch_size
                totals[name]["mf_mae"] += mf_mae * batch_size
                totals[name]["grad_norm"] += grad_norm * batch_size
                totals[name]["n"] += batch_size

        for name, values in totals.items():
            n = max(values["n"], 1.0)
            rows.append(
                {
                    "variant": name,
                    "timestep": t_value,
                    "physics_mse": values["physics_mse"] / n,
                    "mf_mae": values["mf_mae"] / n,
                    "grad_norm": values["grad_norm"] / n,
                }
            )

    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "timestep_error_gradient.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["variant", "timestep", "physics_mse", "mf_mae", "grad_norm"])
        writer.writeheader()
        writer.writerows(rows)
    plot_error_gradient(rows, output_dir / "timestep_error_gradient.png")
    return rows


def plot_error_gradient(rows: list[dict[str, float | int | str]], output_path: Path) -> None:
    labels = {
        "engineer_time": "Engineer entrenado con t",
        "clean_xt": "Clean sobre x_t ruidosa",
    }
    colors = {"engineer_time": "#0072B2", "clean_xt": "#D55E00"}
    fig, axes = plt.subplots(2, 1, figsize=(9, 8), sharex=True)
    for variant, label in labels.items():
        subset = [row for row in rows if row["variant"] == variant]
        xs = np.asarray([row["timestep"] for row in subset], dtype=np.float64)
        mf_mae = np.asarray([row["mf_mae"] for row in subset], dtype=np.float64)
        grad_norm = np.asarray([row["grad_norm"] for row in subset], dtype=np.float64)
        axes[0].plot(xs, mf_mae, marker="o", linewidth=2.0, label=label, color=colors[variant])
        axes[1].plot(xs, grad_norm, marker="o", linewidth=2.0, label=label, color=colors[variant])

    axes[0].set_ylabel("MF MAE")
    axes[0].set_title("Error vs timestep")
    axes[1].set_ylabel("||d objective / d x_t||")
    axes[1].set_xlabel("timestep t")
    axes[1].set_title("Norma del gradiente vs timestep")
    for axis in axes:
        axis.grid(alpha=0.3)
        axis.legend()
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def guidance_weight(args: argparse.Namespace, index: int, total_steps: int) -> float:
    return bell_guidance_weight(
        step_index=index,
        total_steps=total_steps,
        w_max=float(args.bell_w_max),
        peak=float(args.bell_peak),
        width=float(args.bell_width),
    )


def guided_gradient(
    variant: str,
    engineer: LoadedModel,
    clean: LoadedModel,
    architect: torch.nn.Module,
    architect_scheduler: Any,
    eval_stats: dict[str, Any],
    x: torch.Tensor,
    fz_real: torch.Tensor,
    timestep: torch.Tensor,
    model_output: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, float]]:
    if variant == "engineer_time":
        x_req = x.detach().clone().requires_grad_(True)
        pred_norm = model_prediction(engineer, x_req, fz_real, timestep, eval_stats)
        mf_map, objective = mf_and_objective(engineer, pred_norm)
        grad = torch.autograd.grad(objective, x_req, retain_graph=False, create_graph=False)[0]
        return grad, {"objective": float(objective.item()), "mf_mean": float(mf_map.mean().item())}

    if variant == "clean_xt":
        x_req = x.detach().clone().requires_grad_(True)
        t_zero = torch.zeros_like(timestep)
        pred_norm = model_prediction(clean, x_req, fz_real, t_zero, eval_stats)
        mf_map, objective = mf_and_objective(clean, pred_norm)
        grad = torch.autograd.grad(objective, x_req, retain_graph=False, create_graph=False)[0]
        return grad, {"objective": float(objective.item()), "mf_mean": float(mf_map.mean().item())}

    if variant == "clean_tweedie_x0":
        x_req = x.detach().clone().requires_grad_(True)
        with torch.no_grad():
            arch_out = architect(x_req, timestep).sample
        x0_hat = predict_x0_from_model_output(architect_scheduler, x_req, arch_out, timestep)
        t_zero = torch.zeros_like(timestep)
        pred_norm = model_prediction(clean, x0_hat, fz_real, t_zero, eval_stats)
        mf_map, objective = mf_and_objective(clean, pred_norm)
        grad = torch.autograd.grad(objective, x_req, retain_graph=False, create_graph=False)[0]
        return grad, {"objective": float(objective.item()), "mf_mean": float(mf_map.mean().item())}

    raise ValueError(f"Unsupported guidance variant: {variant}")


def run_guided_single_sample(
    engineer: LoadedModel,
    clean: LoadedModel,
    architect_path: Path,
    loader: DataLoader,
    args: argparse.Namespace,
    device: torch.device,
    output_dir: Path,
) -> dict[str, Any]:
    architect_ckpt = load_checkpoint(architect_path, device)
    architect = build_unet(architect_ckpt["model_config"]).to(device)
    architect.load_state_dict(architect_ckpt["model_state_dict"])
    architect.eval()
    architect_scheduler = build_scheduler(architect_ckpt["model_config"])
    architect_scheduler.set_timesteps(int(args.sample_steps), device=device)

    batch = next(iter(loader))
    fz_real = batch["fz_real"][:1].to(device)
    arch_stats = architect_ckpt["normalization_stats"]
    eng_stats = engineer.stats
    fz_real_for_output = fz_real.detach().cpu().numpy()

    variants = ["engineer_time", "clean_xt", "clean_tweedie_x0"]
    histories: dict[str, dict[str, list[float]]] = {}
    samples: dict[str, np.ndarray] = {}

    for variant in variants:
        seed_everything(int(args.seed))
        generator = torch.Generator(device=device)
        generator.manual_seed(int(args.seed))
        x = torch.randn((1, 1, 64, 64), generator=generator, device=device)
        history = {"t": [], "objective": [], "mf_mean": [], "grad_norm": [], "guide_weight": []}

        for index, timestep in enumerate(tqdm(architect_scheduler.timesteps, desc=f"E_DANI guide {variant}")):
            t_value = int(timestep.item())
            t_batch = torch.full((1,), t_value, device=device, dtype=torch.long)
            with torch.no_grad():
                model_output = architect(x, t_batch).sample

            grad, log_values = guided_gradient(
                variant=variant,
                engineer=engineer,
                clean=clean,
                architect=architect,
                architect_scheduler=architect_scheduler,
                eval_stats=arch_stats,
                x=x,
                fz_real=fz_real,
                timestep=t_batch,
                model_output=model_output,
            )
            weight = guidance_weight(args, index, len(architect_scheduler.timesteps))
            grad = torch.clamp(
                float(args.guidance_scale) * weight * grad,
                min=-float(args.guidance_clip),
                max=float(args.guidance_clip),
            )
            alpha_bar_t = architect_scheduler.alphas_cumprod[timestep].to(device)
            guided_output = model_output + torch.sqrt(1.0 - alpha_bar_t) * grad
            x = architect_scheduler.step(guided_output, timestep, x).prev_sample

            history["t"].append(float(t_value))
            history["objective"].append(float(log_values["objective"]))
            history["mf_mean"].append(float(log_values["mf_mean"]))
            history["grad_norm"].append(float(grad.flatten(1).norm(dim=1).mean().item()))
            history["guide_weight"].append(float(weight))

        z_real = minmax_denorm(x.detach(), float(arch_stats["z_min"]), float(arch_stats["z_max"]))
        samples[variant] = z_real.cpu().numpy()
        histories[variant] = history

    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_dir / "guided_single_sample.npz", fz=fz_real_for_output, **samples)
    with (output_dir / "guided_histories.json").open("w", encoding="utf-8") as handle:
        json.dump(histories, handle, indent=2)
    plot_guided_samples(samples, output_dir / "guided_single_sample.png")
    plot_guided_histories(histories, output_dir / "guided_histories.png")
    return {"variants": variants, "sample_npz": str(output_dir / "guided_single_sample.npz")}


def plot_guided_samples(samples: dict[str, np.ndarray], output_path: Path) -> None:
    titles = {
        "engineer_time": "Engineer con t",
        "clean_xt": "Clean sobre x_t",
        "clean_tweedie_x0": "Clean sobre Tweedie x0",
    }
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    values = [array[0, 0] for array in samples.values()]
    vmin = min(float(value.min()) for value in values)
    vmax = max(float(value.max()) for value in values)
    for axis, (variant, array) in zip(axes, samples.items(), strict=False):
        axis.imshow(array[0, 0], cmap="viridis", vmin=vmin, vmax=vmax)
        axis.set_title(titles[variant])
        axis.axis("off")
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_guided_histories(histories: dict[str, dict[str, list[float]]], output_path: Path) -> None:
    labels = {
        "engineer_time": "Engineer con t",
        "clean_xt": "Clean sobre x_t",
        "clean_tweedie_x0": "Clean sobre Tweedie x0",
    }
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharex=True)
    for variant, history in histories.items():
        t = np.asarray(history["t"], dtype=np.float64)
        axes[0].plot(t, history["mf_mean"], label=labels[variant])
        axes[1].plot(t, history["objective"], label=labels[variant])
        axes[2].plot(t, history["grad_norm"], label=labels[variant])
    axes[0].set_title("MF medio")
    axes[1].set_title("Objetivo")
    axes[2].set_title("Gradiente guiado")
    for axis in axes:
        axis.set_xlabel("timestep t")
        axis.invert_xaxis()
        axis.grid(alpha=0.3)
        axis.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    seed_everything(int(args.seed))
    device = resolve_device(str(args.device))
    config = load_config(args.config)
    output_dir = args.output_dir.resolve()

    clean_path = resolve_clean_checkpoint(args.clean_checkpoint.resolve())
    engineer = load_surrogate("engineer_time", args.engineer_checkpoint.resolve(), device)
    clean = load_surrogate("clean", clean_path, device)
    loader = build_validation_loader(config, engineer.stats, int(args.eval_batch_size))

    output_dir.mkdir(parents=True, exist_ok=True)
    evaluate_timestep_curves(engineer, clean, loader, args, device, output_dir)
    guide_summary = run_guided_single_sample(
        engineer=engineer,
        clean=clean,
        architect_path=args.architect_checkpoint.resolve(),
        loader=loader,
        args=args,
        device=device,
        output_dir=output_dir,
    )

    summary = {
        "engineer_checkpoint": str(args.engineer_checkpoint.resolve()),
        "clean_checkpoint": str(clean_path),
        "architect_checkpoint": str(args.architect_checkpoint.resolve()),
        "output_dir": str(output_dir),
        "timestep_stride": int(args.timestep_stride),
        "guidance_scale": float(args.guidance_scale),
        "guide_summary": guide_summary,
    }
    with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
