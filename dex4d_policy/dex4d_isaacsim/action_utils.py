"""Dex4D's action law, engine agnostic. `tasks/fr3_xhand_ap2ap.py:1280-1289`.

Only the non-OSC branch is ported. `useOSCControl` is False in every shipped
config, so the OSC branch and the jacobian and mass-matrix tensors it needs are
dead and do not come across.

Three things carried across deliberately.

The arm integrates on the previous TARGET, not on the measured joint position,
so target error does not feed back into the accumulator.

The hand is an absolute map from [-1, 1] onto the joint limits, so it is not an
accumulator at all. Arm and hand are different control laws in one action vector.

`actionsMovingAverage` is 1.0 in every shipped config, which makes the EMA the
identity. It is kept because it is in the source, not because it does anything.
"""
from __future__ import annotations

import torch


def scale(x: torch.Tensor, lower: torch.Tensor, upper: torch.Tensor) -> torch.Tensor:
    """isaacgym.torch_utils.scale, [-1, 1] onto [lower, upper]."""
    return 0.5 * (x + 1.0) * (upper - lower) + lower


def tensor_clamp(x: torch.Tensor, lower: torch.Tensor, upper: torch.Tensor) -> torch.Tensor:
    return torch.max(torch.min(x, upper), lower)


def compute_targets(actions: torch.Tensor, prev_targets: torch.Tensor,
                    dof_lower: torch.Tensor, dof_upper: torch.Tensor,
                    num_arm_dofs: int, dof_speed_scale: float, dt: float,
                    act_moving_average: float) -> torch.Tensor:
    """Returns cur_targets. Arm accumulates, hand maps absolutely, then EMA and clamp.

    The limits arrive per env, shape (n, num_dofs), because domain randomization
    shifts them per env, `yaml:155-166`. Gym's are a flat (num_dofs,) because it
    caches them once and never refreshes, so its joint slice is `[num_arm_dofs:]`
    where ours has to be `[:, num_arm_dofs:]`. Slicing the env axis by mistake is
    silent until the shapes stop matching.
    """
    cur = torch.empty_like(prev_targets)
    cur[:, :num_arm_dofs] = (prev_targets[:, :num_arm_dofs]
                             + dof_speed_scale * dt * actions[:, :num_arm_dofs])
    cur[:, num_arm_dofs:] = scale(actions[:, num_arm_dofs:],
                                  dof_lower[..., num_arm_dofs:],
                                  dof_upper[..., num_arm_dofs:])
    cur = act_moving_average * cur + (1.0 - act_moving_average) * prev_targets
    return tensor_clamp(cur, dof_lower, dof_upper)
