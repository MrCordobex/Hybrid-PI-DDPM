from __future__ import annotations

from typing import Tuple

import numpy as np
import torch

from tfm_shells.constants import DISPLACEMENT_KEYS, FLEXION_KEYS, MEMBRANE_KEYS


def denormalize_physics(
    pred_norm: torch.Tensor,
    p_mean: torch.Tensor,
    p_std: torch.Tensor,
) -> torch.Tensor:
    return pred_norm * p_std + p_mean


def split_physics_channels(physics_real: torch.Tensor) -> dict[str, torch.Tensor]:
    idx = 0
    uz = physics_real[:, idx : idx + len(DISPLACEMENT_KEYS)]
    idx += len(DISPLACEMENT_KEYS)

    membrane = physics_real[:, idx : idx + len(MEMBRANE_KEYS)]
    idx += len(MEMBRANE_KEYS)

    flexion = physics_real[:, idx : idx + len(FLEXION_KEYS)]
    return {"uz": uz, "membrane": membrane, "flexion": flexion}


def branchwise_supervised_losses(
    pred_norm: torch.Tensor,
    target_norm: torch.Tensor,
) -> dict[str, torch.Tensor]:
    pred = split_physics_channels(pred_norm)
    target = split_physics_channels(target_norm)
    return {
        "uz_mse": torch.mean((pred["uz"] - target["uz"]) ** 2),
        "membrane_mse": torch.mean((pred["membrane"] - target["membrane"]) ** 2),
        "flexion_mse": torch.mean((pred["flexion"] - target["flexion"]) ** 2),
    }


def compute_energy_terms_from_real_physics(
    physics_real: torch.Tensor,
    ds: torch.Tensor,
    dv: torch.Tensor,
    fz: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    branches = split_physics_channels(physics_real)
    uz = branches["uz"]
    membrane = branches["membrane"]
    flexion = branches["flexion"]

    se11, se22, se12 = membrane[:, 0:1], membrane[:, 1:2], membrane[:, 2:3]
    sf11, sf22, sf12 = membrane[:, 3:4], membrane[:, 4:5], membrane[:, 5:6]
    sk11, sk22, sk12 = flexion[:, 0:1], flexion[:, 1:2], flexion[:, 2:3]
    sm11, sm22, sm12 = flexion[:, 3:4], flexion[:, 4:5], flexion[:, 5:6]

    w_memb = (sf11 * se11 + sf22 * se22 + 2.0 * sf12 * se12) * ds
    w_flex = (sm11 * sk11 + sm22 * sk22 + 2.0 * sm12 * sk12) * ds
    w_ext = fz * uz * dv
    return w_memb, w_flex, w_ext


def compute_physical_residual(
    pred_norm: torch.Tensor,
    p_mean: torch.Tensor,
    p_std: torch.Tensor,
    ds: torch.Tensor,
    dv: torch.Tensor,
    fz: torch.Tensor,
) -> torch.Tensor:
    physics_real = denormalize_physics(pred_norm, p_mean, p_std)
    w_memb, w_flex, w_ext = compute_energy_terms_from_real_physics(physics_real, ds, dv, fz)
    delta_p = w_memb.sum(dim=(1, 2, 3)) + w_flex.sum(dim=(1, 2, 3)) - w_ext.sum(dim=(1, 2, 3))
    return delta_p.square()


def compute_membrane_factor_map_from_real_physics(
    physics_real: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    branches = split_physics_channels(physics_real)
    membrane = branches["membrane"]
    flexion = branches["flexion"]

    se11, se22, se12 = membrane[:, 0:1], membrane[:, 1:2], membrane[:, 2:3]
    sf11, sf22, sf12 = membrane[:, 3:4], membrane[:, 4:5], membrane[:, 5:6]
    sk11, sk22, sk12 = flexion[:, 0:1], flexion[:, 1:2], flexion[:, 2:3]
    sm11, sm22, sm12 = flexion[:, 3:4], flexion[:, 4:5], flexion[:, 5:6]

    w_memb = sf11 * se11 + sf22 * se22 + 2.0 * sf12 * se12
    w_flex = sm11 * sk11 + sm22 * sk22 + 2.0 * sm12 * sk12
    mf = w_memb / (w_memb + w_flex + eps)
    return torch.clamp(mf, 0.0, 1.0)


def compute_membrane_factor_from_prediction(
    pred_norm: torch.Tensor,
    p_mean: torch.Tensor,
    p_std: torch.Tensor,
) -> torch.Tensor:
    physics_real = denormalize_physics(pred_norm, p_mean, p_std)
    return compute_membrane_factor_map_from_real_physics(physics_real)


def compute_mf_mean_numpy(mf_map: np.ndarray) -> float:
    return float(np.asarray(mf_map).mean())
