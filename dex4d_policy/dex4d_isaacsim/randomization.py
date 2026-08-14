"""Domain randomization.

Schedules are clocked on the simulate frame count, not the policy step count.
Gravity randomization is not implemented, IsaacLab fixes gravity at construction.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import torch


@dataclass
class NoiseSpec:
    rng: tuple = (0.0, 0.0)
    rng_corr: tuple = (0.0, 0.0)
    schedule_steps: int = 40000


@dataclass
class ScaleSpec:
    rng: tuple = (1.0, 1.0)
    schedule_steps: int = 30000
    num_buckets: int = 0


@dataclass
class AdditiveSpec:
    rng: tuple = (0.0, 0.0)
    schedule_steps: int = 30000


@dataclass
class RandomizationCfg:
    enabled: bool = True
    frequency: int = 720                       # sim frames

    observations: NoiseSpec = field(default_factory=lambda: NoiseSpec(
        (0.0, 0.002), (0.0, 0.001), 40000))
    actions: NoiseSpec = field(default_factory=lambda: NoiseSpec(
        (0.0, 0.05), (0.0, 0.015), 40000))

    robot_damping: ScaleSpec = field(default_factory=lambda: ScaleSpec((0.9, 1.1), 30000))
    robot_stiffness: ScaleSpec = field(default_factory=lambda: ScaleSpec((0.9, 1.1), 30000))
    robot_lower: AdditiveSpec = field(default_factory=lambda: AdditiveSpec((0.0, 0.01), 30000))
    robot_upper: AdditiveSpec = field(default_factory=lambda: AdditiveSpec((0.0, 0.01), 30000))
    robot_mass: ScaleSpec = field(default_factory=lambda: ScaleSpec((0.5, 1.5), 30000))
    robot_friction: ScaleSpec = field(default_factory=lambda: ScaleSpec((0.7, 1.3), 30000, 250))
    object_mass: ScaleSpec = field(default_factory=lambda: ScaleSpec((0.5, 1.5), 30000))
    object_friction: ScaleSpec = field(default_factory=lambda: ScaleSpec((0.7, 1.3), 30000, 250))


def _sched(spec, frame: int) -> float:
    """Linear ramp on the sim frame count, clamped to 1.0."""
    if spec.schedule_steps <= 0:
        return 1.0
    return min(1.0, float(frame) / float(spec.schedule_steps))


def gaussian_noise(x: torch.Tensor, corr: torch.Tensor, spec: NoiseSpec,
                   frame: int) -> torch.Tensor:
    """Additive white noise plus a per-env correlated draw, both ramped."""
    s = _sched(spec, frame)
    return x + corr * spec.rng_corr[1] * s + torch.randn_like(x) * spec.rng[1] * s


def refresh_correlated(shape, spec: NoiseSpec, device) -> torch.Tensor:
    return torch.randn(shape, device=device)


def _uniform(lo, hi, shape, device, buckets=0):
    u = torch.rand(shape, device=device)
    if buckets:
        # PhysX caps live materials, so friction is drawn on a discrete grid.
        u = torch.floor(u * buckets) / max(1, buckets - 1)
        u = torch.clamp(u, 0.0, 1.0)
    return lo + (hi - lo) * u


def scale_samples(spec: ScaleSpec, n: int, device, frame: int) -> torch.Tensor:
    """Multiplicative factors. Collapses to 1.0 at the start of the schedule."""
    s = _sched(spec, frame)
    raw = _uniform(spec.rng[0], spec.rng[1], (n,), device, spec.num_buckets)
    return 1.0 + (raw - 1.0) * s


def additive_samples(spec: AdditiveSpec, shape, device, frame: int) -> torch.Tensor:
    s = _sched(spec, frame)
    return torch.randn(shape, device=device) * spec.rng[1] * s
