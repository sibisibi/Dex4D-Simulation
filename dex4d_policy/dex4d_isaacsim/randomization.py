"""IsaacGym's `apply_randomizations`, ported.

`task.randomize` is **True** in every shipped FR3 config, so this fires on every
reset in gym and the port had none of it. Source is `hand_base/base_task.py:245-439`
driven by the `randomization_params` tree at `cfg/fr3_xhand_ap2ap_stage_1_2.yaml:112-203`.

Four things about the original that are easy to get wrong and are reproduced here.

The schedule clock is `gym.get_frame_count`, the **simulate** count, not the
policy count. At `controlFrequencyInv: 2` the yaml's 40000 is 20000 policy steps
and 30000 is 15000.

`frequency: 720` gates the non-env parameters on sim frames, and per-actor
parameters additionally need `randomize_buf >= 720` and membership in the reset
set. `first_randomization` overrides both and randomizes everything once.

Observation noise lands on `obs_buf` only, never on `states_buf`, so under the
symmetric teacher the actor sees noise and `get_state` does not.

Action noise is applied AFTER the +-1 clamp, so the stored action can exceed the
range and the arm integrator and the action penalty both see the noisy value.

Not ported, and named rather than dropped in silence. `sim_params.gravity` is a
scene-wide write in gym and IsaacLab exposes gravity only on `SimulationCfg` at
construction, so a per-720-frame rewrite has no equivalent that does not restart
the simulator. Everything else in the tree is here.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import torch


@dataclass
class NoiseSpec:
    """One `range` / `range_correlated` / `schedule_steps` block."""
    rng: tuple = (0.0, 0.0)
    rng_corr: tuple = (0.0, 0.0)
    schedule_steps: int = 40000


@dataclass
class ScaleSpec:
    """One uniform `scaling` block."""
    rng: tuple = (1.0, 1.0)
    schedule_steps: int = 30000
    num_buckets: int = 0


@dataclass
class AdditiveSpec:
    rng: tuple = (0.0, 0.0)
    schedule_steps: int = 30000


@dataclass
class RandomizationCfg:
    """yaml :106-203, value for value."""
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
    """`base_task.py:284-291`. Linear ramp on the SIM frame count, clamped."""
    if spec.schedule_steps <= 0:
        return 1.0
    return min(1.0, float(frame) / float(spec.schedule_steps))


def gaussian_noise(x: torch.Tensor, corr: torch.Tensor, spec: NoiseSpec,
                   frame: int) -> torch.Tensor:
    """`base_task.py:308-316`. Additive, white plus a per-env correlated draw
    that is refreshed on the `frequency` beat, both ramped."""
    s = _sched(spec, frame)
    return x + corr * spec.rng_corr[1] * s + torch.randn_like(x) * spec.rng[1] * s


def refresh_correlated(shape, spec: NoiseSpec, device) -> torch.Tensor:
    return torch.randn(shape, device=device)


def _uniform(lo, hi, shape, device, buckets=0):
    u = torch.rand(shape, device=device)
    if buckets:
        # `dr_utils.py:226`. PhysX caps live materials, so friction is drawn on
        # a discrete grid rather than continuously.
        u = torch.floor(u * buckets) / max(1, buckets - 1)
        u = torch.clamp(u, 0.0, 1.0)
    return lo + (hi - lo) * u


def scale_samples(spec: ScaleSpec, n: int, device, frame: int) -> torch.Tensor:
    """`scaling` with a linear schedule. At schedule 0 the sample collapses to
    1.0, which is `base_task.py:296-300`'s `min(steps, sched)/sched` form."""
    s = _sched(spec, frame)
    raw = _uniform(spec.rng[0], spec.rng[1], (n,), device, spec.num_buckets)
    return 1.0 + (raw - 1.0) * s


def additive_samples(spec: AdditiveSpec, shape, device, frame: int) -> torch.Tensor:
    s = _sched(spec, frame)
    return torch.randn(shape, device=device) * spec.rng[1] * s
