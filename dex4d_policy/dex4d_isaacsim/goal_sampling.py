"""Goal sampling. Position perturbation and rotation composition, pure torch."""
from __future__ import annotations

import torch

XYZ_BOUND = ((-0.3, 1.0), (-0.5, 0.5), (0.65, 1.1))


def quat_mul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """xyzw quaternion product."""
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
    """Perturb by uniform(-dis_range, dis_range) per axis, then clamp to bounds."""
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
