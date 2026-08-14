"""Success, reset and curriculum."""
from __future__ import annotations

import torch

from reward_utils import ACHIEVED_LAST_TIME, SUCCESS_THRESHOLD  # noqa: F401

STAGE2_START_ITERATION_DEFAULT = 15000


def update_achieved(achieved_buf: torch.Tensor,
                    goal_obj_dist: torch.Tensor) -> torch.Tensor:
    """Count consecutive in-tolerance steps, reset to zero on a miss."""
    return torch.where(goal_obj_dist <= SUCCESS_THRESHOLD, achieved_buf + 1,
                       torch.zeros_like(achieved_buf))


def goal_reached(achieved_buf: torch.Tensor) -> torch.Tensor:
    """True after ACHIEVED_LAST_TIME consecutive in-tolerance steps."""
    return achieved_buf >= ACHIEVED_LAST_TIME


def check_termination(progress_buf: torch.Tensor, max_episode_length: int,
                      goal_obj_dist: torch.Tensor, too_far_threshold: float,
                      test: bool = False) -> torch.Tensor:
    """Reset flag as an int. Time-out, plus object-goal distance unless testing."""
    time_out = (progress_buf >= max_episode_length).int()
    if test:
        return time_out
    too_far = (goal_obj_dist >= too_far_threshold).int()
    return time_out + too_far


def update_curriculum(iteration: int, curriculum: bool,
                      stage2_start_iteration: int = STAGE2_START_ITERATION_DEFAULT):
    """Stage 2 overrides, or None before stage2_start_iteration."""
    if not curriculum or iteration < stage2_start_iteration:
        return None
    return dict(too_far_reset_threshold=1e6,
                goal_reset_stable_ratio=0.2,
                robot_dof_speed_scale=1.5)
