"""Scene construction, and the two conventions that differ from the task.

Joint order is IsaacLab's, so every consumer goes through `joint_permutation`.
IsaacLab quaternions are wxyz while the task is xyzw, converted once here.
"""
from __future__ import annotations

import json
from pathlib import Path

import torch


def load_goal_usd_cache(cache_root: str) -> list[str]:
    """The goalviz bake of each entry, collision disabled."""
    m = json.loads((Path(cache_root) / "manifest.json").read_text())
    return [str(Path(cache_root) / e["goal_usd"]) for e in m["entries"]]


def load_usd_cache(cache_root: str, object_cls: str | None = None) -> list[str]:
    """The converted corpus listed by its manifest."""
    root = Path(cache_root)
    manifest = root / "manifest.json"
    if not manifest.is_file():
        raise FileNotFoundError(f"no manifest at {manifest}")
    m = json.loads(manifest.read_text())
    if object_cls and m["cls"] != object_cls:
        raise ValueError(
            f"cache holds cls={m['cls']} but the env wants {object_cls}")
    if m["n_failed"]:
        raise ValueError(f"{m['n_failed']} objects failed conversion, refusing "
                         f"to train on a partial corpus")
    return [str(root / e["usd"]) for e in m["entries"]]


def load_usd_cache_keys(cache_root: str) -> list[str]:
    """`code|scale` per entry, in the same order as `load_usd_cache`."""
    m = json.loads((Path(cache_root) / "manifest.json").read_text())
    return [f"{e['code']}|{e['scale']}" for e in m["entries"]]


def assign_objects_to_envs(usd_paths: list[str], num_envs: int) -> list[int]:
    """Each env holds one object for the whole run, chosen at scene build."""
    if not usd_paths:
        raise ValueError("empty object pool")
    return [i % len(usd_paths) for i in range(num_envs)]


def apply_physx_material_properties(env) -> None:
    """Set friction on every shape of the robot, the table and the object.

    Runs after the simulator is up because `UsdFileCfg` carries no material
    field. The material row is (static, dynamic, restitution).
    """
    f = float(env.d4.scene.friction)
    env_ids = torch.arange(env.num_envs, dtype=torch.int64, device="cpu")

    for name in ("robot", "table", "object"):
        view = getattr(env, name).root_physx_view
        mats = view.get_material_properties()
        mats[:, :, 0] = f
        mats[:, :, 1] = f
        view.set_material_properties(mats, env_ids)


def load_visual_features(root: str, cache_keys: list[str], assignment: list[int],
                         device) -> "torch.Tensor":
    """Per-object PointNet features from `<root>/<code>/pc_feat_<scale>.npy`."""
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
    """canon_to_lab and lab_to_canon, asserted to be inverse bijections."""
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
    """Indexes the last axis, so (N, K, 4) works the same as (N, 4)."""
    return torch.cat([q[..., 1:4], q[..., 0:1]], dim=-1)


def quat_xyzw_to_wxyz(q: torch.Tensor) -> torch.Tensor:
    return torch.cat([q[..., 3:4], q[..., 0:3]], dim=-1)


def build_robot_cfg(usd_path: str, cfg):
    """Arm and hand are separate actuator groups, they carry different gains."""
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
            # Friction is set after launch, see apply_physx_material_properties.
            collision_props=sim_utils.CollisionPropertiesCfg(
                contact_offset=cfg.scene.contact_offset,
                rest_offset=cfg.scene.rest_offset),
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
    """A kinematic cuboid."""
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
    """The object, or the goal ghost, which is kinematic with gravity off.

    The ghost's collision is disabled in its baked USD, not here. Setting
    `collision_enabled` on the spawn cfg does not reach nested collision prims.
    """
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
