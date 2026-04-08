from __future__ import annotations

from typing import Any

import torch
from diffusers import DDPMScheduler, UNet2DModel

from tfm_shells.models.parallel_pb_unet import ParallelPBUNet


def build_unet(model_config: dict[str, Any]) -> torch.nn.Module:
    kind = str(model_config.get("kind", "unet"))
    if kind == "parallel_pb_unet":
        return ParallelPBUNet(
            sample_size=int(model_config["sample_size"]),
            in_channels=int(model_config["in_channels"]),
            layers_per_block=int(model_config["layers_per_block"]),
            block_out_channels=tuple(int(value) for value in model_config["block_out_channels"]),
            down_block_types=tuple(model_config["down_block_types"]),
            up_block_types=tuple(model_config["up_block_types"]),
            branch_channels=model_config.get("branch_channels"),
        )
    return UNet2DModel(
        sample_size=int(model_config["sample_size"]),
        in_channels=int(model_config["in_channels"]),
        out_channels=int(model_config["out_channels"]),
        layers_per_block=int(model_config["layers_per_block"]),
        block_out_channels=tuple(int(value) for value in model_config["block_out_channels"]),
        down_block_types=tuple(model_config["down_block_types"]),
        up_block_types=tuple(model_config["up_block_types"]),
    )


def build_scheduler(model_config: dict[str, Any]) -> DDPMScheduler:
    return DDPMScheduler(
        num_train_timesteps=1000,
        beta_schedule=str(model_config["beta_schedule"]),
        prediction_type=str(model_config["prediction_type"]),
    )


def count_parameters(model: torch.nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())
