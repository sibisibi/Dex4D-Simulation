"""Action law. Arm integrates on its previous target, hand maps absolutely."""
from __future__ import annotations

import torch


def scale(x: torch.Tensor, lower: torch.Tensor, upper: torch.Tensor) -> torch.Tensor:
    """Map [-1, 1] onto [lower, upper]."""
    return 0.5 * (x + 1.0) * (upper - lower) + lower


def tensor_clamp(x: torch.Tensor, lower: torch.Tensor, upper: torch.Tensor) -> torch.Tensor:
    return torch.max(torch.min(x, upper), lower)


def compute_targets(actions: torch.Tensor, prev_targets: torch.Tensor,
                    dof_lower: torch.Tensor, dof_upper: torch.Tensor,
                    num_arm_dofs: int, dof_speed_scale: float, dt: float,
                    act_moving_average: float) -> torch.Tensor:
    """Arm accumulates, hand maps absolutely, then EMA and clamp.

    dof_lower and dof_upper are per env, shape (n, num_dofs), since randomization
    shifts the limits per env.
    """
    cur = torch.empty_like(prev_targets)
    cur[:, :num_arm_dofs] = (prev_targets[:, :num_arm_dofs]
                             + dof_speed_scale * dt * actions[:, :num_arm_dofs])
    cur[:, num_arm_dofs:] = scale(actions[:, num_arm_dofs:],
                                  dof_lower[..., num_arm_dofs:],
                                  dof_upper[..., num_arm_dofs:])
    cur = act_moving_average * cur + (1.0 - act_moving_average) * prev_targets
    return tensor_clamp(cur, dof_lower, dof_upper)
