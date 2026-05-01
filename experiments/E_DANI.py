from __future__ import annotations

import argparse
import csv
import gc
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
DEFAULT_CLEAN = PROJECT_ROOT / "models" / "engineer" / "modelo_clean" / "best.pt"
LEGACY_CLEAN = PROJECT_ROOT / "models" / "modelo_clean" / "best.pt"
TYPO_CLEAN = PROJECT_ROOT / "models" / "enginieers" / "modelo_clean" / "best.pt"
DEFAULT_ARCHITECT = PROJECT_ROOT / "models" / "architect" / "latest" / "best.pt"
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "engineer.yaml"


@dataclass(frozen=True)
class LoadedModel:
    name: str
    model: torch.nn.Module
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
    parser.add_argument("--eval-batch-size", type=int, default=16)
    parser.add_argument("--eval-max-batches", type=int, default=None)
    parser.add_argument("--sample-steps", type=int, default=1000)
    parser.add_argument("--guidance-scale", type=float, default=10.0)
    parser.add_argument("--guidance-clip", type=float, default=5.0)
    parser.add_argument("--bell-w-max", type=float, default=1.0)
    parser.add_argument("--bell-peak", type=float, default=0.5)
    parser.add_argument("--bell-width", type=float, default=0.22)
    return parser.parse_args()


def log(message: str) -> None:
    print(f"[E_DANI] {message}", flush=True)


def format_gb(num_bytes: int) -> str:
    return f"{num_bytes / (1024 ** 3):.2f} GB"


def as_project_path(path: Path) -> Path:
    path = Path(path).expanduser()
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def load_checkpoint(path: Path, label: str, device: torch.device) -> dict[str, Any]:
    path = as_project_path(path)
    if not path.exists():
        raise FileNotFoundError(f"{label} checkpoint not found: {path}")
    stat = path.stat()
    log(f"{label}: torch.load START | path={path} | size={format_gb(stat.st_size)} | map_location={device}")
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    keys = ", ".join(sorted(str(key) for key in checkpoint.keys()))
    log(f"{label}: torch.load DONE | keys=[{keys}]")
    return checkpoint


def resolve_clean_checkpoint(path: Path) -> Path:
    path = as_project_path(path)
    if path.exists():
        return path
    for candidate in (DEFAULT_CLEAN, LEGACY_CLEAN, TYPO_CLEAN):
        if candidate.exists():
            log(f"clean checkpoint requested path missing; using existing candidate: {candidate}")
            return candidate
    raise FileNotFoundError(f"Clean checkpoint not found: {path}")


def load_surrogate(
    name: str,
    checkpoint_path: Path,
    device: torch.device,
) -> LoadedModel:
    checkpoint = load_checkpoint(checkpoint_path, label=name, device=device)
    model_config = checkpoint["model_config"]
    stats = checkpoint["normalization_stats"]
    log(f"{name}: build_unet START | kind={model_config.get('kind', 'unet')} | device={device}")
    model = build_unet(model_config).to(device)
    log(f"{name}: build_unet DONE")
    log(f"{name}: load_state_dict START")
    state_dict = checkpoint["model_state_dict"]
    model.load_state_dict(state_dict)
    log(f"{name}: load_state_dict DONE")
    model.eval()
    del checkpoint, state_dict
    gc.collect()
    return LoadedModel(
        name=name,
        model=model,
        scheduler=build_scheduler(model_config),
        stats=stats,
        include_fz=int(model_config.get("in_channels", 1)) > 1,
    )


def load_architect(
    checkpoint_path: Path,
    device: torch.device,
) -> tuple[torch.nn.Module, Any, dict[str, Any]]:
    checkpoint = load_checkpoint(checkpoint_path, label="architect", device=device)
    model_config = checkpoint["model_config"]
    stats = checkpoint["normalization_stats"]
    log(f"architect: build_unet START | kind={model_config.get('kind', 'unet')} | device={device}")
    model = build_unet(model_config).to(device)
    log("architect: build_unet DONE")
    log("architect: load_state_dict START")
    state_dict = checkpoint["model_state_dict"]
    model.load_state_dict(state_dict)
    log("architect: load_state_dict DONE")
    model.eval()
    scheduler = build_scheduler(model_config)
    del checkpoint, state_dict
    gc.collect()
    return model, scheduler, stats


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
    log(f"indexing validation dataset: {dataset_dir}")
    records = build_dataset_index(dataset_dir)
    filtered = filter_records(
        records,
        subset=str(config["data"]["subset"]),
        min_mf_mean=config["data"].get("min_mf_mean"),
    )
    _, val_records = split_records(filtered, val_ratio=float(config["data"]["val_ratio"]), seed=int(config["seed"]))
    log(f"validation records: filtered={len(filtered)} | val={len(val_records)} | batch_size={batch_size}")
    dataset = ShellDataset(val_records, stats=stats, include_physics=True)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=False)


def tweedie_prediction_error_and_grad(
    clean: LoadedModel,
    architect: torch.nn.Module,
    architect_scheduler: Any,
    architect_stats: dict[str, Any],
    x_t: torch.Tensor,
    fz_real: torch.Tensor,
    physics_real_target: torch.Tensor,
    mf_true: torch.Tensor,
    timesteps: torch.Tensor,
    eval_stats: dict[str, Any],
) -> tuple[float, float, float]:
    """Standard DPS-style baseline: Tweedie-denoise x_t through the
    Architect, then evaluate the clean surrogate on the resulting
    x0_hat at t=0. The gradient w.r.t. x_t is taken end-to-end through
    the analytical Tweedie identity (the Architect output is detached,
    matching the standard practical implementation of DPS)."""
    x_req = x_t.detach().clone().requires_grad_(True)
    with torch.no_grad():
        arch_out = architect(x_req, timesteps).sample
    x0_hat = predict_x0_from_model_output(architect_scheduler, x_req, arch_out, timesteps)
    t_zero = torch.zeros_like(timesteps)
    pred_norm = model_prediction(clean, x0_hat, fz_real, t_zero, eval_stats)
    p_mean, p_std = expand_physics_stats(clean.stats, pred_norm.shape[0], pred_norm.device)
    pred_real = pred_norm * p_std + p_mean
    mf_pred, objective = mf_and_objective(clean, pred_norm)
    grad = torch.autograd.grad(objective, x_req, retain_graph=False, create_graph=False)[0]

    physics_mse = F.mse_loss(pred_real, physics_real_target).item()
    mf_mae = torch.abs(mf_pred - mf_true).mean().item()
    grad_norm = grad.flatten(1).norm(dim=1).mean().item()
    return float(physics_mse), float(mf_mae), float(grad_norm)


def evaluate_timestep_curves(
    engineer: LoadedModel,
    clean: LoadedModel,
    architect: torch.nn.Module,
    architect_scheduler: Any,
    architect_stats: dict[str, Any],
    loader: DataLoader,
    args: argparse.Namespace,
    device: torch.device,
    output_dir: Path,
) -> list[dict[str, float | int | str]]:
    """Experiment 1. Empirical validation of Proposition 1 (Jensen's
    Gap). Three variants are evaluated under identical noise samples:

      * ``engineer_time``: the time-conditioned Engineer evaluated on
        x_t with timestep t. Predicts E[R(x_0) | x_t] by construction
        and should remain accurate across the full noise spectrum.
      * ``clean_tweedie_x0``: the standard DPS baseline. The Architect
        Tweedie-denoises x_t to x0_hat, on which the clean surrogate
        is evaluated at t=0. It computes R(E[x_0 | x_t]), incurring
        the Jensen's-Gap bias; expected to degrade smoothly as t
        grows because Tweedie's posterior variance grows with t.
      * ``clean_xt``: the most naive option. Feeds the noisy x_t
        directly into a surrogate trained only on clean inputs at
        t=0. Has no mathematical justification and is expected to
        explode catastrophically with t.

    For each variant we report the prediction error on the full
    physics tensor (``physics_mse``), the per-pixel error on the
    Membrane Factor map (``mf_mae``), and the norm of the data-space
    gradient of the funicular objective with respect to x_t
    (``grad_norm``). The latter is the quantity that is actually
    injected into the reverse process during physics-guided sampling
    and is therefore the most operationally meaningful diagnostic of
    the Jensen's Gap."""
    t_max = int(engineer.scheduler.config.num_train_timesteps)
    timesteps = list(range(0, t_max, int(args.timestep_stride)))
    if timesteps[-1] != t_max - 1:
        timesteps.append(t_max - 1)
    log(
        "timestep evaluation config | "
        f"num_timesteps={len(timesteps)} | stride={args.timestep_stride} | "
        f"batch_size={args.eval_batch_size} | max_batches={args.eval_max_batches} | "
        f"num_val_batches={len(loader)}"
    )

    variant_names = ("engineer_time", "clean_tweedie_x0", "clean_xt")
    rows: list[dict[str, float | int | str]] = []
    generator = torch.Generator(device=device)
    generator.manual_seed(int(args.seed))

    for t_value in tqdm(timesteps, desc="E_DANI eval timesteps"):
        log(f"eval timestep START: t={t_value}")
        totals: dict[str, dict[str, float]] = {
            name: {"physics_mse": 0.0, "mf_mae": 0.0, "grad_norm": 0.0, "n": 0.0}
            for name in variant_names
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
                "clean_tweedie_x0": tweedie_prediction_error_and_grad(
                    clean,
                    architect,
                    architect_scheduler,
                    architect_stats,
                    x_t,
                    fz_real,
                    physics_real,
                    mf_true,
                    t_batch,
                    engineer.stats,
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
        log(f"eval timestep DONE: t={t_value}")

    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "timestep_error_gradient.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["variant", "timestep", "physics_mse", "mf_mae", "grad_norm"])
        writer.writeheader()
        writer.writerows(rows)
    log(f"timestep metrics written: {csv_path}")
    plot_error_gradient(rows, output_dir / "timestep_error_gradient.png")
    log(f"timestep figure written: {output_dir / 'timestep_error_gradient.png'}")
    return rows


def plot_error_gradient(rows: list[dict[str, float | int | str]], output_path: Path) -> None:
    labels = {
        "engineer_time": "Time-conditioned Engineer (proposed)",
        "clean_tweedie_x0": "Clean surrogate on Tweedie $\\hat{x}_0$ (DPS)",
        "clean_xt": "Clean surrogate on $x_t$ (naive)",
    }
    colors = {
        "engineer_time": "#0072B2",
        "clean_tweedie_x0": "#E69F00",
        "clean_xt": "#D55E00",
    }
    markers = {
        "engineer_time": "o",
        "clean_tweedie_x0": "s",
        "clean_xt": "^",
    }
    fig, axes = plt.subplots(3, 1, figsize=(9, 11), sharex=True)
    for variant, label in labels.items():
        subset = [row for row in rows if row["variant"] == variant]
        if not subset:
            continue
        subset = sorted(subset, key=lambda row: row["timestep"])
        xs = np.asarray([row["timestep"] for row in subset], dtype=np.float64)
        physics_mse = np.asarray([row["physics_mse"] for row in subset], dtype=np.float64)
        mf_mae = np.asarray([row["mf_mae"] for row in subset], dtype=np.float64)
        grad_norm = np.asarray([row["grad_norm"] for row in subset], dtype=np.float64)
        axes[0].plot(xs, physics_mse, marker=markers[variant], linewidth=2.0, label=label, color=colors[variant])
        axes[1].plot(xs, mf_mae, marker=markers[variant], linewidth=2.0, label=label, color=colors[variant])
        axes[2].plot(xs, grad_norm, marker=markers[variant], linewidth=2.0, label=label, color=colors[variant])

    axes[0].set_ylabel("Physics tensor MSE")
    axes[0].set_title("Surrogate accuracy across the diffusion noise spectrum")
    axes[0].set_yscale("log")
    axes[1].set_ylabel("Membrane Factor MAE")
    axes[1].set_title("Per-pixel Membrane Factor error")
    axes[2].set_ylabel(r"$\|\nabla_{x_t}\, J(x_t)\|_2$")
    axes[2].set_xlabel("Diffusion timestep $t$")
    axes[2].set_title("Norm of the guidance gradient injected into the reverse step")
    axes[2].set_yscale("log")
    for axis in axes:
        axis.grid(alpha=0.3, which="both")
        axis.legend(frameon=False)
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
    architect: torch.nn.Module,
    architect_scheduler: Any,
    arch_stats: dict[str, Any],
    loader: DataLoader,
    args: argparse.Namespace,
    device: torch.device,
    output_dir: Path,
) -> dict[str, Any]:
    """Experiment 2. End-to-end physics-guided sampling under three
    gradient providers, all sharing the same Architect, the same bell
    schedule, the same global scale and the same random seed, so that
    any difference between the resulting trajectories is attributable
    purely to the gradient quality:

      * ``engineer_time`` (proposed) -- time-conditioned Engineer.
      * ``clean_tweedie_x0`` (DPS baseline) -- clean surrogate on
        Tweedie x0_hat at t=0; standard recipe in physics-guided
        diffusion that incurs Jensen's-Gap bias.
      * ``clean_xt`` (naive) -- clean surrogate fed directly with the
        noisy x_t at t=0; included as the most pedagogical
        illustration of the bias."""
    architect_scheduler.set_timesteps(int(args.sample_steps), device=device)
    log(
        "guided sampling config | "
        f"sample_steps={args.sample_steps} | guidance_scale={args.guidance_scale} | "
        f"guidance_clip={args.guidance_clip}"
    )

    batch = next(iter(loader))
    fz_real = batch["fz_real"][:1].to(device)
    fz_real_for_output = fz_real.detach().cpu().numpy()

    variants = ["engineer_time", "clean_xt", "clean_tweedie_x0"]
    histories: dict[str, dict[str, list[float]]] = {}
    samples: dict[str, np.ndarray] = {}

    for variant in variants:
        log(f"guided variant START: {variant}")
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
        log(f"guided variant DONE: {variant}")

    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_dir / "guided_single_sample.npz", fz=fz_real_for_output, **samples)
    log(f"guided samples written: {output_dir / 'guided_single_sample.npz'}")
    with (output_dir / "guided_histories.json").open("w", encoding="utf-8") as handle:
        json.dump(histories, handle, indent=2)
    log(f"guided histories written: {output_dir / 'guided_histories.json'}")
    plot_guided_samples(samples, output_dir / "guided_single_sample.png")
    plot_guided_histories(histories, output_dir / "guided_histories.png")
    log(f"guided figures written: {output_dir / 'guided_single_sample.png'} and {output_dir / 'guided_histories.png'}")
    return {"variants": variants, "sample_npz": str(output_dir / "guided_single_sample.npz")}


def plot_guided_samples(samples: dict[str, np.ndarray], output_path: Path) -> None:
    titles = {
        "engineer_time": "Time-conditioned Engineer (proposed)",
        "clean_tweedie_x0": "Clean on Tweedie $\\hat{x}_0$ (DPS)",
        "clean_xt": "Clean on $x_t$ (naive)",
    }
    ordered_variants = [variant for variant in titles if variant in samples]
    fig, axes = plt.subplots(1, len(ordered_variants), figsize=(4 * len(ordered_variants), 4))
    if len(ordered_variants) == 1:
        axes = [axes]
    values = [samples[variant][0, 0] for variant in ordered_variants]
    vmin = min(float(value.min()) for value in values)
    vmax = max(float(value.max()) for value in values)
    for axis, variant in zip(axes, ordered_variants, strict=False):
        axis.imshow(samples[variant][0, 0], cmap="viridis", vmin=vmin, vmax=vmax)
        axis.set_title(titles[variant])
        axis.axis("off")
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_guided_histories(histories: dict[str, dict[str, list[float]]], output_path: Path) -> None:
    labels = {
        "engineer_time": "Time-conditioned Engineer (proposed)",
        "clean_tweedie_x0": "Clean on Tweedie $\\hat{x}_0$ (DPS)",
        "clean_xt": "Clean on $x_t$ (naive)",
    }
    colors = {
        "engineer_time": "#0072B2",
        "clean_tweedie_x0": "#E69F00",
        "clean_xt": "#D55E00",
    }
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharex=True)
    for variant, history in histories.items():
        if variant not in labels:
            continue
        t = np.asarray(history["t"], dtype=np.float64)
        axes[0].plot(t, history["mf_mean"], label=labels[variant], color=colors[variant], linewidth=2.0)
        axes[1].plot(t, history["objective"], label=labels[variant], color=colors[variant], linewidth=2.0)
        axes[2].plot(t, history["grad_norm"], label=labels[variant], color=colors[variant], linewidth=2.0)
    axes[0].set_title("Mean Membrane Factor along the reverse process")
    axes[0].set_ylabel(r"$\overline{\mathrm{mf}}_t$")
    axes[1].set_title("Funicular objective")
    axes[1].set_ylabel(r"$J_t = (1 - \overline{\mathrm{mf}}_t)^2$")
    axes[2].set_title("Norm of the injected gradient")
    axes[2].set_ylabel(r"$\|\widetilde{g}_t\|_2$")
    axes[2].set_yscale("log")
    for axis in axes:
        axis.set_xlabel("Diffusion timestep $t$")
        axis.invert_xaxis()
        axis.grid(alpha=0.3, which="both")
        axis.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    seed_everything(int(args.seed))
    device = resolve_device(str(args.device))
    config = load_config(args.config)
    output_dir = args.output_dir.resolve()

    log(f"START | device={device} | seed={args.seed}")
    log(f"config={args.config}")
    clean_path = resolve_clean_checkpoint(args.clean_checkpoint)
    engineer_path = as_project_path(args.engineer_checkpoint)
    architect_path = as_project_path(args.architect_checkpoint)
    log(f"engineer_checkpoint={engineer_path}")
    log(f"clean_checkpoint={clean_path}")
    log(f"architect_checkpoint={architect_path}")
    engineer = load_surrogate(
        "engineer_time",
        engineer_path,
        device,
    )
    clean = load_surrogate(
        "clean",
        clean_path,
        device,
    )
    architect, architect_scheduler, arch_stats = load_architect(
        architect_path,
        device=device,
    )
    log("validation loader START")
    loader = build_validation_loader(config, engineer.stats, int(args.eval_batch_size))
    log("validation loader DONE")

    output_dir.mkdir(parents=True, exist_ok=True)
    log("timestep evaluation START")
    evaluate_timestep_curves(
        engineer=engineer,
        clean=clean,
        architect=architect,
        architect_scheduler=architect_scheduler,
        architect_stats=arch_stats,
        loader=loader,
        args=args,
        device=device,
        output_dir=output_dir,
    )
    log("timestep evaluation DONE")
    log("guided single sample START")
    guide_summary = run_guided_single_sample(
        engineer=engineer,
        clean=clean,
        architect=architect,
        architect_scheduler=architect_scheduler,
        arch_stats=arch_stats,
        loader=loader,
        args=args,
        device=device,
        output_dir=output_dir,
    )
    log("guided single sample DONE")

    summary = {
        "engineer_checkpoint": str(engineer_path),
        "clean_checkpoint": str(clean_path),
        "architect_checkpoint": str(architect_path),
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
