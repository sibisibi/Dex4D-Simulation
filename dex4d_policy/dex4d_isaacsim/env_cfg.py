"""Task configuration."""
from __future__ import annotations

from dataclasses import dataclass, field

from randomization import RandomizationCfg
from robots import (ARM_ARMATURE, ARM_FRICTION, ARM_GAIN_ROWS, ARM_JOINTS,
                    BASE_POS, HAND_ARMATURE, HAND_DAMPING, HAND_JOINTS,
                    HAND_STIFFNESS, RESET_ARM_POS, RESET_HAND_POS, TABLE_DIMS)


@dataclass
class SceneCfg:
    base_pos: tuple = BASE_POS
    base_quat_xyzw: tuple = (0.0, 0.0, 0.0, 1.0)
    table_dims: tuple = TABLE_DIMS
    table_pos: tuple = (0.0, 0.0, 0.3)         # centre, at half height
    object_spawn: tuple = (0.0, 0.0, 0.70)     # table top plus 0.10
    delta_x_range: tuple = (-0.1, 0.1)
    delta_y_range: tuple = (-0.1, 0.1)
    object_density: float = 500.0
    friction: float = 1.0
    contact_offset: float = 0.002
    rest_offset: float = 0.0
    angular_damping: float = 0.01
    linear_damping: float = 0.01
    ground_z: float = 0.0
    ground_friction: float = 1.0


@dataclass
class GoalCfg:
    displacement: tuple = (0.0, 0.0, 0.2)      # the opening lift
    walk_step: float = 0.1
    walk_angle: float = 0.5                    # one-sided radians
    stable_ratio: float = 0.1                  # curriculum raises it to 0.2
    xyz_bound: tuple = ((-0.3, 1.0), (-0.5, 0.5), (0.65, 1.1))


@dataclass
class RewardCfg:
    obj_finger: float = 1.0
    obj_hand: float = 1.0
    goal_obj: float = 1.0
    success_bonus: float = 1.0
    terminal_bonus: float = 1.0
    finger_curl_reg: float = 1.0
    table_collision_penalty: float = 1.0
    fall_penalty: float = 0.0
    hand_vel_penalty: float = 0.0
    dof_vel: float = 0.0
    dof_acc: float = 0.0
    action_penalty: float = 1.0

    def as_dict(self) -> dict:
        return {k: getattr(self, k) for k in self.__dataclass_fields__}


@dataclass
class TerminationCfg:
    episode_length: int = 400
    too_far_reset_threshold: float = 0.3       # becomes 1e6 after stage2_start_iteration
    success_threshold: float = 0.05
    achieved_last_time: int = 30
    object_init_stable_ratio: float = 0.2


@dataclass
class ControlCfg:
    num_arm_dofs: int = 7
    num_hand_dofs: int = 12
    dof_speed_scale: float = 10.0              # curriculum drops it to 1.5
    act_moving_average: float = 1.0
    control_frequency_inv: int = 4             # decimation, 30 Hz control
    sim_dt: float = 1.0 / 120.0
    # The action law's dt is one policy step, not the physics step.
    action_dt: float = 1.0 / 60.0
    use_osc_control: bool = False


@dataclass
class PushCfg:
    """Overwrites every object's velocity on a fixed interval."""
    enabled: bool = True
    interval_s: float = 4.0
    max_linvel_xy: float = 0.2
    max_angvel: float = 0.2


@dataclass
class ObsCfg:
    num_keypoints: int = 128
    kp_downsample_ratio: int = 2
    num_fingertips: int = 5
    asymmetric_observations: bool = False
    clip_observations: float = 5.0


@dataclass
class PhysxCfg:
    substeps: int = 2
    solver_type: int = 1
    num_position_iterations: int = 8
    num_velocity_iterations: int = 0
    contact_offset: float = 0.002
    rest_offset: float = 0.0
    bounce_threshold_velocity: float = 0.2
    max_depenetration_velocity: float = 1000.0


@dataclass
class Dex4DEnvCfg:
    num_envs: int = 4096
    env_spacing: float = 3.0                   # GridCloner pitch, not a half extent
    arm_gain_row: str = "a3"
    object_cls: str = "bottle"                 # None for every class
    usd_cache: str = ""                        # the converted corpus
    corpus_root: str = ""                      # source meshes, for the viewer
    visual_feat_root: str = ""                 # the 64-dim observation block
    robot_urdf_src: str = ""                   # robot urdf, for the viewer
    curriculum: bool = True
    stage2_start_iteration: int = 15000
    random_time: bool = True                   # one-shot episode phase stagger

    scene: SceneCfg = field(default_factory=SceneCfg)
    goal: GoalCfg = field(default_factory=GoalCfg)
    reward: RewardCfg = field(default_factory=RewardCfg)
    termination: TerminationCfg = field(default_factory=TerminationCfg)
    control: ControlCfg = field(default_factory=ControlCfg)
    obs: ObsCfg = field(default_factory=ObsCfg)
    physx: PhysxCfg = field(default_factory=PhysxCfg)
    push: PushCfg = field(default_factory=PushCfg)
    randomization: RandomizationCfg = field(default_factory=RandomizationCfg)

    def arm_gains(self):
        return ARM_GAIN_ROWS[self.arm_gain_row]

    def stage3(self) -> "Dex4DEnvCfg":
        """Every class at 5 Hz, contact terms off, heavier regularisation."""
        import copy
        c = copy.deepcopy(self)
        c.object_cls = None
        c.control.control_frequency_inv = 24
        c.reward.obj_finger = 0.0
        c.reward.obj_hand = 0.0
        c.reward.finger_curl_reg = 10.0
        c.reward.action_penalty = 5.0
        return c


ARM_JOINT_NAMES = ARM_JOINTS
HAND_JOINT_NAMES = HAND_JOINTS
RESET_POS = RESET_ARM_POS + RESET_HAND_POS
ACTUATOR_ARMATURE = ARM_ARMATURE + HAND_ARMATURE
ACTUATOR_FRICTION = ARM_FRICTION + (0.0,) * 12
HAND_KP = HAND_STIFFNESS
HAND_KD = HAND_DAMPING
