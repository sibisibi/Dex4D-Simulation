"""Object and goal reset. Simulator writes are left to the caller."""
from __future__ import annotations

import torch

from goal_sampling import sample_position, sample_rotation


def quat_from_euler_xyz(roll, pitch, yaw):
    """Returns xyzw."""
    cy, sy = (yaw * 0.5).cos(), (yaw * 0.5).sin()
    cr, sr = (roll * 0.5).cos(), (roll * 0.5).sin()
    cp, sp = (pitch * 0.5).cos(), (pitch * 0.5).sin()
    return torch.stack([
        cy * sr * cp - sy * cr * sp,
        cy * cr * sp + sy * sr * cp,
        sy * cr * cp - cy * sr * sp,
        cy * cr * cp + sy * sr * sp], dim=-1)


def reset_object_pose(n, object_init_state, delta_x_range, delta_y_range,
                      object_init_stable_ratio, generator=None, device="cpu"):
    """Returns (pos, quat xyzw). z is untouched, only x and y jitter."""
    def u(lo, hi, shape):
        return torch.rand(shape, generator=generator, device=device) * (hi - lo) + lo

    theta = u(-3.14, 3.14, (n,))
    euler_xy = u(-3.14, 3.14, (n, 2))
    dx = u(delta_x_range[0], delta_x_range[1], (n, 1))
    dy = u(delta_y_range[0], delta_y_range[1], (n, 1))

    stable = u(0.0, 1.0, (n,)) < object_init_stable_ratio
    roll = torch.where(stable, torch.full_like(theta, torch.pi / 2.0), euler_xy[:, 0])
    pitch = torch.where(stable, torch.zeros_like(theta), euler_xy[:, 1])

    quat = quat_from_euler_xyz(roll, pitch, theta)
    pos = object_init_state[:, 0:3].clone()
    pos[:, 0:1] += dx
    pos[:, 1:2] += dy
    return pos, quat


def reset_goal_pose_init(object_pos, object_rot, goal_displacement):
    """The opening goal. Pure translation, orientation copied, no clamp."""
    return object_pos + goal_displacement, object_rot.clone()


def reset_goal_pose_walk(object_pos, object_rot, object_init_state,
                         goal_reset_stable_ratio, dis_range=0.1, angle_range=0.5,
                         bounds=None, generator=None, device="cpu"):
    """A later goal. Anchored on the object, clamped, then a fraction forced flat."""
    n = object_pos.shape[0]
    kw = dict(bounds=bounds) if bounds is not None else {}
    pos = sample_position(object_pos, dis_range, **kw)
    rot = sample_rotation(object_rot, angle_range)

    r = torch.rand(n, generator=generator, device=device)
    flat = (r < goal_reset_stable_ratio).nonzero(as_tuple=False).squeeze(-1)
    if flat.numel():
        theta = (torch.rand(flat.numel(), generator=generator, device=device)
                 * 6.28 - 3.14)
        # After the clamp, and it is the spawn height not the table top.
        pos[flat, 2] = object_init_state[flat, 2]
        rot[flat] = quat_from_euler_xyz(
            torch.full((flat.numel(),), torch.pi / 2.0, device=device),
            torch.zeros(flat.numel(), device=device), theta)
    return pos, rot, flat
