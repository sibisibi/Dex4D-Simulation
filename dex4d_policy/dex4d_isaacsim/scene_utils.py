"""Dex4D's scene, built the way SimToolReal's IsaacLab backend builds its own.

Their `scene_utils.py` is the one genuinely new file in a gym-to-lab port, and
this is its Dex4D counterpart. The differences from theirs are the ones the
corpus forces.

  they    generate object URDFs procedurally per launch, convert per launch
  here    the corpus is fixed and already converted offline, so this reads a
          prebuilt USD cache and round-robins it, which is what MultiUsdFileCfg
          already does

The robot USD is converted once per launch, same as theirs, because it is one
asset and the conversion is under a second.

Two things the port must not lose, both from SimToolReal's own invariants test.
Joint ordering is IsaacLab's, not IsaacGym's, so `joint_permutation` builds the
map and every consumer goes through it. And IsaacLab quaternions are wxyz while
Dex4D's whole task is xyzw, so the conversion happens once at the boundary here
rather than being sprinkled through the task math.
"""
from __future__ import annotations

import json
from pathlib import Path

import torch


def load_goal_usd_cache(cache_root: str) -> list[str]:
    """The goalviz bake of each entry, collision disabled.

    Gym isolates the goal with collision group `i + num_envs` so it touches
    nothing. Setting `collision_enabled=False` on the spawn cfg does not reach
    the nested collision prims, so the flag is baked into a separate USD, which
    is the route SimToolReal takes for the same reason.
    """
    m = json.loads((Path(cache_root) / "manifest.json").read_text())
    return [str(Path(cache_root) / e["goal_usd"]) for e in m["entries"]]


def load_usd_cache(cache_root: str, object_cls: str | None = None) -> list[str]:
    """The manifest the offline converter wrote. Fails loud on a miss."""
    root = Path(cache_root)
    manifest = root / "manifest.json"
    if not manifest.is_file():
        raise FileNotFoundError(
            f"no manifest at {manifest}. Run convert_corpus.py first, it takes "
            f"85 s for the bottle set.")
    m = json.loads(manifest.read_text())
    if object_cls and m["cls"] != object_cls:
        raise ValueError(
            f"cache holds cls={m['cls']} but the env wants {object_cls}. "
            f"Convert the right set rather than filtering a wrong one.")
    if m["n_failed"]:
        raise ValueError(f"{m['n_failed']} objects failed conversion, refusing "
                         f"to train on a partial corpus")
    return [str(root / e["usd"]) for e in m["entries"]]


def load_usd_cache_keys(cache_root: str) -> list[str]:
    """`code|scale` per entry, in the same order as `load_usd_cache`.

    Read from the manifest rather than parsed back out of the USD path, because
    the converter nests the stem twice and path parsing was silently wrong.
    """
    m = json.loads((Path(cache_root) / "manifest.json").read_text())
    return [f"{e['code']}|{e['scale']}" for e in m["entries"]]


def assign_objects_to_envs(usd_paths: list[str], num_envs: int) -> list[int]:
    """Each env holds ONE object for the whole run, chosen at scene build. That
    is what Dex4D does too, `object_code_and_scale_str_for_envs` is fixed at
    construction, so the round-robin is faithful rather than a simplification."""
    if not usd_paths:
        raise ValueError("empty object pool")
    return [i % len(usd_paths) for i in range(num_envs)]


def apply_physx_material_properties(env) -> None:
    """tasks:726-736, reproducing gym's resulting physics, not its literal reach.

    Gym writes friction on every robot shape but only on `table_shape_props[0]`
    and `object_shape_props[0]`. Copying that reach would be wrong here because
    the DEFAULTS differ. IsaacGym's shape friction already defaults to 1.0, so
    gym's narrow write is a no-op and the whole object ends at 1.0 anyway.
    IsaacLab defaults to 0.5, so a shape-0-only write would leave the other
    sixteen convex pieces at half gym's friction. The goal ghost is untouched on
    both sides, and it now has no collision shapes at all.

    `UsdFileCfg` has no `physics_material`, so this goes through the PhysX tensor
    views after the simulator is up, which is the route SimToolReal takes in its
    own `apply_physx_material_properties` for the same reason. The material row
    is (static, dynamic, restitution) and gym leaves restitution alone.
    """
    f = float(env.d4.scene.friction)
    env_ids = torch.arange(env.num_envs, dtype=torch.int64, device="cpu")

    # Every shape of the robot, the table and the object. Gym writes only
    # `object_shape_props[0]`, but copying that reach is wrong here because the
    # DEFAULTS differ. IsaacGym's shape friction defaults to 1.0, so gym's write
    # is a no-op and the whole object ends at 1.0. IsaacLab's defaults to 0.5,
    # so a shape-0-only write would leave every other convex piece at half
    # gym's friction. Writing 1.0 everywhere is what reproduces gym's physics.
    for name in ("robot", "table", "object"):
        view = getattr(env, name).root_physx_view
        mats = view.get_material_properties()
        mats[:, :, 0] = f
        mats[:, :, 1] = f
        view.set_material_properties(mats, env_ids)


def load_visual_features(root: str, cache_keys: list[str], assignment: list[int],
                         device) -> "torch.Tensor":
    """The 64-dim per-object PointNet feature, tasks:489.

    Path is `meshdatav3_pc_feat/<code>/pc_feat_<scale>.npy` and the manifest
    already carries both halves as `code|scale`. Crashes on a missing file
    rather than handing the policy a zero row, because a zero row is exactly the
    bug this replaces.
    """
    import numpy as np

    cache = {}
    rows = []
    for e_i in assignment:
        key = cache_keys[e_i]
        if key not in cache:
            code, scale = key.split("|")
            cache[key] = np.load(Path(root) / code / f"pc_feat_{scale}.npy")
        rows.append(cache[key])
    return torch.tensor(np.stack(rows), device=device, dtype=torch.float32)


def joint_permutation(lab_joint_names: list[str],
                      canonical: tuple) -> tuple[torch.Tensor, torch.Tensor]:
    """canon_to_lab and lab_to_canon, asserted to be inverse bijections.

    This is the single most common way a gym-to-lab port goes silently wrong,
    and it is the first thing SimToolReal's invariants test checks.
    """
    if sorted(lab_joint_names) != sorted(canonical):
        missing = set(canonical) - set(lab_joint_names)
        extra = set(lab_joint_names) - set(canonical)
        raise ValueError(f"joint sets differ. missing {missing}, extra {extra}")
    lab_index = {n: i for i, n in enumerate(lab_joint_names)}
    canon_to_lab = torch.tensor([lab_index[n] for n in canonical], dtype=torch.long)
    lab_to_canon = torch.empty_like(canon_to_lab)
    lab_to_canon[canon_to_lab] = torch.arange(len(canonical))
    assert torch.equal(canon_to_lab[lab_to_canon],
                       torch.arange(len(canonical))), "permutations are not inverses"
    return canon_to_lab, lab_to_canon


def quat_wxyz_to_xyzw(q: torch.Tensor) -> torch.Tensor:
    """IsaacLab is wxyz, Dex4D's task math is xyzw. Converted once, here.

    Indexes the last axis, so a (N, K, 4) batch of fingertip quaternions works
    the same as a flat (N, 4). Slicing dim 1 silently took the wrong columns.
    """
    return torch.cat([q[..., 1:4], q[..., 0:1]], dim=-1)


def quat_xyzw_to_wxyz(q: torch.Tensor) -> torch.Tensor:
    return torch.cat([q[..., 3:4], q[..., 0:3]], dim=-1)


def build_robot_cfg(usd_path: str, cfg):
    """ArticulationCfg mirroring SimToolReal's, with Dex4D's own values.

    The arm and hand are separate actuator groups because they carry different
    gains, armature and friction, and because the action law treats them
    differently, the arm accumulates and the hand maps absolutely.
    """
    from isaaclab.actuators import ImplicitActuatorCfg
    from isaaclab.assets import ArticulationCfg
    from isaaclab.sim import UsdFileCfg
    import isaaclab.sim as sim_utils

    from env_cfg import ARM_JOINT_NAMES, HAND_JOINT_NAMES, HAND_KD, HAND_KP
    from robots import (ARM_ARMATURE, ARM_FRICTION, ARM_GAIN_ROWS, HAND_ARMATURE,
                        RESET_ARM_POS, RESET_HAND_POS)

    kp, kd = ARM_GAIN_ROWS[cfg.arm_gain_row]
    joint_pos = {n: p for n, p in zip(ARM_JOINT_NAMES, RESET_ARM_POS)}
    joint_pos.update({n: p for n, p in zip(HAND_JOINT_NAMES, RESET_HAND_POS)})

    return ArticulationCfg(
        prim_path="/World/envs/env_.*/Robot",
        spawn=UsdFileCfg(
            usd_path=usd_path, activate_contact_sensors=False,
            # The cfg sim block's two offsets, per collider in lab. Friction is
            # NOT set here, `UsdFileCfg` has no `physics_material` field, so it
            # goes through PhysX views after launch, see
            # `apply_physx_material_properties`.
            collision_props=sim_utils.CollisionPropertiesCfg(
                contact_offset=cfg.scene.contact_offset,
                rest_offset=cfg.scene.rest_offset),
            # tasks:396-397. Gym cuts angular damping 50x below the IsaacGym
            # default and raises linear damping off zero, deliberately, and the
            # converter's own values are neither.
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                angular_damping=cfg.scene.angular_damping,
                linear_damping=cfg.scene.linear_damping)),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=tuple(cfg.scene.base_pos),
            rot=tuple(quat_xyzw_to_wxyz(
                torch.tensor([cfg.scene.base_quat_xyzw]))[0].tolist()),
            joint_pos=joint_pos,
            joint_vel={".*": 0.0},
        ),
        actuators={
            "arm": ImplicitActuatorCfg(
                joint_names_expr=list(ARM_JOINT_NAMES),
                stiffness={n: v for n, v in zip(ARM_JOINT_NAMES, kp)},
                damping={n: v for n, v in zip(ARM_JOINT_NAMES, kd)},
                armature={n: v for n, v in zip(ARM_JOINT_NAMES, ARM_ARMATURE)},
                friction={n: v for n, v in zip(ARM_JOINT_NAMES, ARM_FRICTION)},
            ),
            "hand": ImplicitActuatorCfg(
                joint_names_expr=list(HAND_JOINT_NAMES),
                stiffness={n: v for n, v in zip(HAND_JOINT_NAMES, HAND_KP)},
                damping={n: v for n, v in zip(HAND_JOINT_NAMES, HAND_KD)},
                armature={n: v for n, v in zip(HAND_JOINT_NAMES, HAND_ARMATURE)},
            ),
        },
    )


def build_table_cfg(cfg):
    """Dex4D builds the table with `create_box`, so a cuboid spawn is the
    faithful counterpart rather than a URDF."""
    from isaaclab.assets import RigidObjectCfg
    import isaaclab.sim as sim_utils

    dx, dy, dz = cfg.scene.table_dims
    return RigidObjectCfg(
        prim_path="/World/envs/env_.*/Table",
        spawn=sim_utils.CuboidCfg(
            size=(dx, dy, dz),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            collision_props=sim_utils.CollisionPropertiesCfg(
                contact_offset=cfg.scene.contact_offset,
                rest_offset=cfg.scene.rest_offset),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=cfg.scene.friction,
                dynamic_friction=cfg.scene.friction),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=tuple(cfg.scene.table_pos)),
    )


def build_object_cfg(usd_paths: list[str], cfg, goal_viz: bool,
                     goal_usd_paths: list[str] | None = None):
    """The object, or the goal ghost. The ghost is kinematic with gravity off and
    collision disabled, matching Dex4D's `goal_asset_options`."""
    from isaaclab.assets import RigidObjectCfg
    import isaaclab.sim as sim_utils
    from isaaclab.sim.spawners.wrappers import MultiUsdFileCfg

    props = sim_utils.RigidBodyPropertiesCfg(
        kinematic_enabled=goal_viz,
        disable_gravity=goal_viz,
        max_depenetration_velocity=cfg.physx.max_depenetration_velocity,
    )
    spawn = MultiUsdFileCfg(
        usd_path=list(goal_usd_paths if goal_viz else usd_paths), random_choice=False,
        rigid_props=props,
        # `collision_enabled` defaults to None, which IsaacLab documents as "not
        # modified", so the converted USD's collision survives. Gym isolates the
        # goal with collision group `i + num_envs` and it touches nothing. Left
        # enabled, the ghost is a kinematic collider 0.20 m directly above the
        # object at every reset, in the hand's way. SimToolReal bakes False on
        # its goalviz for the same reason.
        collision_props=sim_utils.CollisionPropertiesCfg(
            contact_offset=cfg.scene.contact_offset,
            rest_offset=cfg.scene.rest_offset),
        mass_props=None if goal_viz else sim_utils.MassPropertiesCfg(density=cfg.scene.object_density),
    )
    name = "GoalViz" if goal_viz else "Object"
    z = cfg.scene.object_spawn[2] + (cfg.goal.displacement[2] if goal_viz else 0.0)
    return RigidObjectCfg(
        prim_path=f"/World/envs/env_.*/{name}",
        spawn=spawn,
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(cfg.scene.object_spawn[0], cfg.scene.object_spawn[1], z)),
    )
