from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch

from tfm_shells.config import load_config, resolve_project_path
from tfm_shells.models.factory import build_scheduler, build_unet
from tfm_shells.training.common import (
    bell_guidance_weight,
    make_run_name,
    polynomial_guidance_weight,
    prepare_run_directories,
    resolve_device,
    seed_everything,
)
from tfm_shells.utils.io import save_json
from tfm_shells.utils.physics import compute_membrane_factor_map_from_real_physics
from tfm_shells.utils.tracking import ExperimentTracker


def _normalize_minmax(array: np.ndarray, minimum: float, maximum: float) -> np.ndarray:
    scale = maximum - minimum
    if abs(scale) < 1e-12:
        return np.zeros_like(array, dtype=np.float32)
    return (2.0 * ((array - minimum) / scale) - 1.0).astype(np.float32)


def _renormalize_tensor(
    x_norm: torch.Tensor,
    src_min: float,
    src_max: float,
    dst_min: float,
    dst_max: float,
) -> torch.Tensor:
    x_real = ((x_norm + 1.0) / 2.0) * (src_max - src_min) + src_min
    return 2.0 * ((x_real - dst_min) / (dst_max - dst_min + 1e-8)) - 1.0


def _load_checkpoint(path: Path, device: torch.device) -> dict[str, Any]:
    return torch.load(path, map_location=device, weights_only=False)


def _bell_or_poly(config: dict[str, Any], step_index: int, total_steps: int) -> float:
    sampling_cfg = config["sampling"]
    mode = str(sampling_cfg["guidance_schedule"])
    if mode == "bell":
        return bell_guidance_weight(
            step_index=step_index,
            total_steps=total_steps,
            w_max=float(sampling_cfg["guide_w_max"]),
            peak=float(sampling_cfg["bell_peak"]),
            width=float(sampling_cfg["bell_width"]),
        )
    return polynomial_guidance_weight(
        step_index=step_index,
        total_steps=total_steps,
        w_min=float(sampling_cfg["guide_w_min"]),
        w_max=float(sampling_cfg["guide_w_max"]),
        power=float(sampling_cfg["guide_power"]),
    )


def run_guided_sampling(config_path: str | Path) -> dict[str, Any]:
    config = load_config(config_path)
    seed_everything(int(config["seed"]))
    directories = prepare_run_directories(config, role="sample")
    runtime_cfg = config.get("runtime", {})
    device = resolve_device(str(runtime_cfg.get("device", "auto")))

    architect_ckpt = _load_checkpoint(resolve_project_path(config, config["architect"]["checkpoint"]), device)
    engineer_ckpt = _load_checkpoint(resolve_project_path(config, config["engineer"]["checkpoint"]), device)

    architect = build_unet(architect_ckpt["model_config"]).to(device)
    architect.load_state_dict(architect_ckpt["model_state_dict"])
    architect.eval()

    engineer = build_unet(engineer_ckpt["model_config"]).to(device)
    engineer.load_state_dict(engineer_ckpt["model_state_dict"])
    engineer.eval()

    scheduler = build_scheduler(architect_ckpt["model_config"])
    scheduler.set_timesteps(int(config["sampling"]["num_inference_steps"]), device=device)

    source_file = resolve_project_path(config, config["conditioning"]["source_file"])
    with np.load(source_file) as data:
        fz_real = data["fz"].astype(np.float32)

    arch_stats = architect_ckpt["normalization_stats"]
    eng_stats = engineer_ckpt["normalization_stats"]
    fz_norm_arch = _normalize_minmax(fz_real, float(arch_stats["fz_min"]), float(arch_stats["fz_max"]))
    fz_norm_eng = _normalize_minmax(fz_real, float(eng_stats["fz_min"]), float(eng_stats["fz_max"]))
    fz_cond_arch = torch.from_numpy(fz_norm_arch).unsqueeze(0).repeat(int(config["conditioning"]["batch_size"]), 1, 1, 1).to(device)
    fz_cond_eng = torch.from_numpy(fz_norm_eng).unsqueeze(0).repeat(int(config["conditioning"]["batch_size"]), 1, 1, 1).to(device)

    x = torch.randn((fz_cond_arch.shape[0], 1, 64, 64), device=device)
    history = {"t": [], "objective": [], "mf_mean": [], "grad_norm": [], "guide_weight": []}

    p_mean = torch.tensor(eng_stats["physics_mean"], dtype=torch.float32, device=device).unsqueeze(0)
    p_std = torch.tensor(eng_stats["physics_std"], dtype=torch.float32, device=device).unsqueeze(0)

    for index, timestep in enumerate(scheduler.timesteps):
        t_value = int(timestep.item())
        t_batch = torch.full((x.shape[0],), t_value, device=device, dtype=torch.long)

        with torch.no_grad():
            architect_input = torch.cat([x, fz_cond_arch], dim=1) if int(architect_ckpt["model_config"]["in_channels"]) > 1 else x
            model_output = architect(architect_input, t_batch).sample

        x_req = x.detach().clone().requires_grad_(True)
        x_req_eng = _renormalize_tensor(
            x_req,
            src_min=float(arch_stats["z_min"]),
            src_max=float(arch_stats["z_max"]),
            dst_min=float(eng_stats["z_min"]),
            dst_max=float(eng_stats["z_max"]),
        )
        engineer_input = torch.cat([x_req_eng, fz_cond_eng], dim=1) if int(engineer_ckpt["model_config"]["in_channels"]) > 1 else x_req_eng
        pred_phys_norm = engineer(engineer_input, t_batch).sample
        pred_phys_real = pred_phys_norm * p_std + p_mean
        mf_map = compute_membrane_factor_map_from_real_physics(pred_phys_real)
        mf_mean = mf_map.mean(dim=(1, 2, 3))
        objective = ((1.0 - mf_mean) ** 2).mean()

        grad = torch.autograd.grad(objective, x_req, retain_graph=False, create_graph=False)[0]
        guide_weight = _bell_or_poly(config, index, len(scheduler.timesteps))
        grad = torch.clamp(
            guide_weight * float(config["sampling"]["guidance_scale"]) * grad,
            min=-float(config["sampling"]["grad_clip"]),
            max=float(config["sampling"]["grad_clip"]),
        )

        alpha_bar_t = scheduler.alphas_cumprod[timestep].to(device)
        sqrt_one_minus_alpha_bar = torch.sqrt(1.0 - alpha_bar_t)
        guided_output = model_output + sqrt_one_minus_alpha_bar * grad
        x = scheduler.step(guided_output, timestep, x).prev_sample

        history["t"].append(t_value)
        history["objective"].append(float(objective.item()))
        history["mf_mean"].append(float(mf_mean.mean().item()))
        history["grad_norm"].append(float(grad.flatten(1).norm(dim=1).mean().item()))
        history["guide_weight"].append(float(guide_weight))

    z_min = float(arch_stats["z_min"])
    z_max = float(arch_stats["z_max"])
    samples_real = ((x.detach().cpu().numpy() + 1.0) / 2.0) * (z_max - z_min) + z_min

    output_npz = directories["run_root"] / "guided_samples.npz"
    np.savez_compressed(output_npz, z=samples_real)

    figure, axes = plt.subplots(2, int(np.ceil(samples_real.shape[0] / 2.0)), figsize=(16, 6))
    axes = np.array(axes).reshape(-1)
    for idx, axis in enumerate(axes):
        axis.axis("off")
        if idx < samples_real.shape[0]:
            axis.imshow(samples_real[idx, 0], cmap="viridis")
            axis.set_title(f"sample_{idx}")
    figure.suptitle("Guided conditional samples")
    figure.tight_layout()
    samples_png = directories["run_root"] / "guided_samples.png"
    figure.savefig(samples_png, dpi=180, bbox_inches="tight")
    plt.close(figure)

    history_png = directories["run_root"] / "guided_history.png"
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    ax[0].plot(history["t"], history["mf_mean"])
    ax[0].invert_xaxis()
    ax[0].set_title("mf_mean")
    ax[0].grid(alpha=0.3)
    ax[1].plot(history["t"], history["objective"])
    ax[1].invert_xaxis()
    ax[1].set_title("objective")
    ax[1].grid(alpha=0.3)
    ax[2].plot(history["t"], history["grad_norm"])
    ax[2].invert_xaxis()
    ax[2].set_title("grad_norm")
    ax[2].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(history_png, dpi=180, bbox_inches="tight")
    plt.close(fig)

    run_name = make_run_name(config, role="sample")
    with ExperimentTracker(config, directories["project_root"], run_name) as tracker:
        tracker.log_config(config)
        tracker.log_metrics(
            {
                "samples_generated": float(samples_real.shape[0]),
                "final_objective": float(history["objective"][-1]),
                "final_mf_mean": float(history["mf_mean"][-1]),
                "final_grad_norm": float(history["grad_norm"][-1]),
            }
        )
        tracker.log_artifact(output_npz, artifact_path="samples")
        tracker.log_artifact(samples_png, artifact_path="samples")
        tracker.log_artifact(history_png, artifact_path="samples")

    summary = {
        "samples_file": str(output_npz),
        "plot_file": str(samples_png),
        "final_mf_mean": float(history["mf_mean"][-1]),
    }
    save_json(summary, directories["run_root"] / "summary.json")
    return summary
