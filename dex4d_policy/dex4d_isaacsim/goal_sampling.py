"""Dex4D's goal sampler, engine agnostic.

Straight translation of `utils/util.py:142` `sample_position` and
`sample_rotation`. Pure torch, no isaacgym and no isaaclab, which is what lets
the parity harness run both this and the gym original in one process.

Two things are carried across deliberately rather than tidied.

The clamp bounds are the same literal Dex4D uses. They are passed as an argument
here because in Dex4D that literal is shared with the stock xArm6 task, and
editing it in place silently retunes that task.

`sample_rotation`'s angle is one sided, `uniform(0, angle_range)`, not two sided.
That is Dex4D's behaviour and it caps the per goal reorientation at 28.6 degrees.
"""
from __future__ import annotations

import torch

# utils/util.py:145. Dex4D's own xyz_bound.
XYZ_BOUND = ((-0.3, 1.0), (-0.5, 0.5), (0.65, 1.1))


def quat_mul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """xyzw, matching isaacgym.torch_utils.quat_mul."""
    x1, y1, z1, w1 = a[:, 0], a[:, 1], a[:, 2], a[:, 3]
    x2, y2, z2, w2 = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
    return torch.stack([
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 + y1 * w2 + z1 * x2 - x1 * z2,
        w1 * z2 + z1 * w2 + x1 * y2 - y1 * x2,
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
    ], dim=-1)


def sample_position(pos: torch.Tensor, dis_range: float,
                    bounds: tuple = XYZ_BOUND) -> torch.Tensor:
    """Perturb by uniform(-dis_range, dis_range) per axis, then clamp per axis.

    The anchor is the object's current position, not the previous goal. That is
    the difference from SimToolReal and it is why the box behaves as a leash on
    a walk the object drags rather than as a workspace.
    """
    b = torch.tensor(bounds, device=pos.device, dtype=pos.dtype)
    noise = torch.empty_like(pos).uniform_(-dis_range, dis_range)
    return torch.clamp(pos + noise, min=b[:, 0], max=b[:, 1])


def sample_rotation(quat: torch.Tensor, angle_range: float) -> torch.Tensor:
    """Compose a random-axis rotation of uniform(0, angle_range) radians. xyzw."""
    axis = torch.randn(quat.shape[0], 3, device=quat.device, dtype=quat.dtype)
    axis = axis / torch.norm(axis, dim=-1, keepdim=True)
    theta = torch.rand(quat.shape[0], device=quat.device, dtype=quat.dtype) * angle_range
    dq_xyz = axis * torch.sin(theta / 2.0).unsqueeze(-1)
    dq_w = torch.cos(theta / 2.0).unsqueeze(-1)
    dq = torch.cat([dq_xyz, dq_w], dim=-1)
    new_quat = quat_mul(dq, quat)
    return new_quat / new_quat.norm(dim=-1, keepdim=True)
