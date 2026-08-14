"""Reward terms, pure torch."""
from __future__ import annotations

import torch

ACHIEVED_LAST_TIME = 30
SUCCESS_THRESHOLD = 0.05
TABLE_HEIGHT_TOLERANCE = 0.62      # table top is 0.60


def contact_flag(obj_finger_dist: torch.Tensor,
                 obj_hand_dist: torch.Tensor) -> torch.Tensor:
    """2 when the fingers and the palm are both near the object."""
    return (obj_finger_dist <= 0.6).int() + (obj_hand_dist <= 0.12).int()


def reward_obj_finger(obj_finger_dist: torch.Tensor) -> torch.Tensor:
    return -0.5 * obj_finger_dist


def reward_obj_hand(obj_hand_dist: torch.Tensor) -> torch.Tensor:
    return -0.5 * obj_hand_dist


def reward_goal_obj(goal_obj_dist: torch.Tensor, flag: torch.Tensor) -> torch.Tensor:
    """A level, not a progress delta. Crosses zero at d = 0.4667."""
    out = torch.zeros_like(goal_obj_dist)
    return torch.where(flag == 2, 1 * (1.4 - 3 * goal_obj_dist), out)


def reward_success_bonus(goal_obj_dist: torch.Tensor, flag: torch.Tensor) -> torch.Tensor:
    out = torch.zeros_like(goal_obj_dist)
    return torch.where(
        flag == 2,
        torch.where(goal_obj_dist <= SUCCESS_THRESHOLD,
                    5.0 / (1 + 10 * goal_obj_dist), out),
        out)


def reward_terminal_bonus(achieved_buf: torch.Tensor) -> torch.Tensor:
    return torch.where(achieved_buf >= ACHIEVED_LAST_TIME, 10.0, 0.0)


def reward_finger_curl_reg(hand_dof_pos: torch.Tensor,
                           curled_q: torch.Tensor) -> torch.Tensor:
    d = (hand_dof_pos - curled_q).norm(p=2, dim=-1)
    return -0.001 * d ** 2


def reward_table_collision_penalty(fingertip_pos: torch.Tensor,
                                   hand_pos: torch.Tensor) -> torch.Tensor:
    """Penalise the lowest fingertip or palm below TABLE_HEIGHT_TOLERANCE."""
    heights = torch.cat([fingertip_pos[:, :, 2], hand_pos[:, 2].unsqueeze(-1)], dim=1)
    lowest, _ = torch.min(heights, dim=1)
    return torch.where(lowest < TABLE_HEIGHT_TOLERANCE,
                       (lowest - TABLE_HEIGHT_TOLERANCE) * 10, torch.zeros_like(lowest))


def reward_fall_penalty(goal_obj_dist: torch.Tensor, too_far: float) -> torch.Tensor:
    return torch.where((goal_obj_dist >= too_far), -5.0, 0.0)


def reward_hand_vel_penalty(hand_vel: torch.Tensor, goal_obj_dist: torch.Tensor,
                            flag: torch.Tensor) -> torch.Tensor:
    out = torch.zeros_like(goal_obj_dist)
    v = torch.sum(torch.square(hand_vel), dim=-1)
    return torch.where(flag == 2,
                       torch.where(goal_obj_dist <= SUCCESS_THRESHOLD, -0.1 * v, out),
                       out)


def reward_dof_vel(dof_vel: torch.Tensor) -> torch.Tensor:
    return torch.clamp(-1e-3 * torch.sum(torch.square(dof_vel), dim=-1), min=-0.5, max=0.0)


def reward_dof_acc(last_dof_vel: torch.Tensor, dof_vel: torch.Tensor,
                   dt: float) -> torch.Tensor:
    dof_acc = (last_dof_vel - dof_vel) / dt
    return torch.clamp(-1e-8 * torch.sum(torch.square(dof_acc), dim=-1), min=-0.5, max=0.0)


def reward_action_penalty(actions: torch.Tensor) -> torch.Tensor:
    return torch.clamp(-0.01 * torch.sum(torch.square(actions), dim=-1), min=-0.5, max=0.0)


IMPLEMENTED_TERMS = (
    "obj_finger", "obj_hand", "goal_obj", "success_bonus", "terminal_bonus",
    "finger_curl_reg", "table_collision_penalty", "fall_penalty",
    "hand_vel_penalty", "dof_vel", "dof_acc", "action_penalty",
)


def active_terms(reward_scales: dict) -> dict:
    """Drop zero-scaled terms rather than multiplying by zero."""
    return {k: v for k, v in reward_scales.items() if v != 0}
