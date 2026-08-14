"""Dex4D's success, reset and curriculum, engine agnostic.

`check_termination` at :1154, the achieved and goal-reset bookkeeping inside
`compute_reward` at :1452, and `update_curriculum` at :1604.

Two things carried across deliberately.

The far reset measures the OBJECT against its GOAL, not the hand against the
object. SimToolReal's equivalent is a hand-wander check and never looks at the
goal. And after `stage2_start_iteration` the threshold becomes 1e6, so for most
of a run there is no far reset at all.

`stage2_start_iteration` defaults to 15000 and appears in no FR3 yaml, so the
default always applies. Stage 3 resumes at 25000, which means its yaml values
for the far reset, the stable ratio and the speed scale are overwritten on the
first call and never used.
"""
from __future__ import annotations

import torch

# One source of truth. Two copies meant a one-sided edit would
# desynchronise the success gate from the success counter.
from reward_utils import ACHIEVED_LAST_TIME, SUCCESS_THRESHOLD  # noqa: F401



STAGE2_START_ITERATION_DEFAULT = 15000


def update_achieved(achieved_buf: torch.Tensor,
                    goal_obj_dist: torch.Tensor) -> torch.Tensor:
    """:1414. Counts consecutive in-tolerance steps, resets to zero on a miss."""
    return torch.where(goal_obj_dist <= SUCCESS_THRESHOLD, achieved_buf + 1,
                       torch.zeros_like(achieved_buf))


def goal_reached(achieved_buf: torch.Tensor) -> torch.Tensor:
    """30 consecutive steps. 1.0 s at 30 Hz, 6.0 s at stage 3's 5 Hz. The paper
    states 0.5 s, so the shipped requirement is twice it and then twelve times
    it, because a step count stayed fixed while the control rate changed."""
    return achieved_buf >= ACHIEVED_LAST_TIME


def check_termination(progress_buf: torch.Tensor, max_episode_length: int,
                      goal_obj_dist: torch.Tensor, too_far_threshold: float,
                      test: bool = False) -> torch.Tensor:
    """:1154. Returns the reset flag as an int, matching the source's arithmetic."""
    time_out = (progress_buf >= max_episode_length).int()
    if test:
        return time_out
    too_far = (goal_obj_dist >= too_far_threshold).int()
    return time_out + too_far


def update_curriculum(iteration: int, curriculum: bool,
                      stage2_start_iteration: int = STAGE2_START_ITERATION_DEFAULT):
    """:1604. Returns the three values it overwrites, or None when it does not fire."""
    if not curriculum or iteration < stage2_start_iteration:
        return None
    return dict(too_far_reset_threshold=1e6,
                goal_reset_stable_ratio=0.2,
                robot_dof_speed_scale=1.5)
