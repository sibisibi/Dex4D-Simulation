"""Every Dex4D constant, as a configclass. Mirrors SimToolReal's
`simtoolreal_env_cfg.py`, which turned the gym monolith's scattered `self.*`
assignments into one readable surface.

Values match `cfg/fr3_xhand_ap2ap_stage_1_2.yaml` and the constants hardcoded in
`tasks/fr3_xhand_ap2ap.py`. Where a yaml key is dead the comment says so, because
silently carrying a dead key across is how a port acquires behaviour nobody asked
for.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from randomization import RandomizationCfg
from robots import (ARM_ARMATURE, ARM_FRICTION, ARM_GAIN_ROWS, ARM_JOINTS,
                    BASE_POS, HAND_ARMATURE, HAND_DAMPING, HAND_JOINTS,
                    HAND_STIFFNESS, RESET_ARM_POS, RESET_HAND_POS, TABLE_DIMS)


@dataclass
class SceneCfg:
    base_pos: tuple = BASE_POS                 # tasks:568
    base_quat_xyzw: tuple = (0.0, 0.0, 0.0, 1.0)   # identity, already faces the table
    table_dims: tuple = TABLE_DIMS             # tasks:123
    table_pos: tuple = (0.0, 0.0, 0.3)         # tasks:580, centre at half height
    object_spawn: tuple = (0.0, 0.0, 0.70)     # tasks:576, table top plus 0.10
    delta_x_range: tuple = (-0.1, 0.1)
    delta_y_range: tuple = (-0.1, 0.1)
    object_density: float = 500.0              # tasks:493
    friction: float = 1.0                      # tasks:722-725, every shape
    contact_offset: float = 0.002              # cfg sim block, per collider in lab
    rest_offset: float = 0.0
    angular_damping: float = 0.01              # tasks:396
    linear_damping: float = 0.01               # tasks:397
    ground_z: float = 0.0                      # gym's PlaneParams default
    ground_friction: float = 1.0               # gym's PlaneParams default


@dataclass
class GoalCfg:
    displacement: tuple = (0.0, 0.0, 0.2)      # tasks:570, the opening lift
    walk_step: float = 0.1                     # sample_position dis_range
    walk_angle: float = 0.5                    # sample_rotation, ONE sided radians
    stable_ratio: float = 0.1                  # stage 1 and 2, curriculum raises to 0.2
    # utils/util.py:145. Passed explicitly because in Dex4D this literal is
    # shared with the stock xArm6 task.
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
    success_threshold: float = 0.05            # HARDCODED at tasks:225
    achieved_last_time: int = 30               # 1.0 s at 30 Hz, 6.0 s at 5 Hz
    object_init_stable_ratio: float = 0.2
    # successTolerance in the yaml is DEAD, read at tasks:47 and never used.


@dataclass
class ControlCfg:
    num_arm_dofs: int = 7
    num_hand_dofs: int = 12
    dof_speed_scale: float = 10.0              # dofSpeedScale, curriculum drops to 1.5
    act_moving_average: float = 1.0            # identity, no smoothing
    # Gym runs sim dt 1/60 with substeps 2, so PhysX integrates at 120 Hz under
    # 30 Hz control. IsaacLab has no substeps, so the same rates come from a
    # 1/120 step with four of them per policy step. SimToolReal's lab cfg makes
    # the identical swap and says so at simtoolreal_env_cfg.py:20.
    control_frequency_inv: int = 4             # decimation, 30 Hz control
    sim_dt: float = 1.0 / 120.0
    # The action law's own constant is gym's `self.dt = sim_params.dt`, which is
    # 1/60 and applied once per policy step. It is NOT the physics step and must
    # not follow it down, or the arm moves at half speed.
    action_dt: float = 1.0 / 60.0
    use_osc_control: bool = False              # the OSC branch is not ported


@dataclass
class PushCfg:
    """tasks:1331 `_push_objects`. Every object's velocity is overwritten with a
    fresh draw on a fixed interval, which is what forces a re-grasp."""
    enabled: bool = True                       # task.push_objects
    interval_s: float = 4.0                    # task.push_interval_s
    max_linvel_xy: float = 0.2
    max_angvel: float = 0.2


@dataclass
class ObsCfg:
    num_keypoints: int = 128
    kp_downsample_ratio: int = 2
    num_fingertips: int = 5
    asymmetric_observations: bool = False      # True only for the DAgger student
    clip_observations: float = 5.0             # never binds, coordinates stay under 1.5


@dataclass
class PhysxCfg:
    """cfg sim block, copied field for field so the two backends step alike."""
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
    # Gym passes `envSpacing` 1.5 as the half extent, `lower = -spacing` and
    # `upper = +spacing`, so IsaacGym pitches env origins by 3.0 m. IsaacLab's
    # GridCloner treats spacing as the pitch itself, so it has to be doubled to
    # keep the same 1.8 m gap between neighbouring 1.2 m tables.
    env_spacing: float = 3.0
    arm_gain_row: str = "a3"                   # a3 is what Dex4D ran before 032
    object_cls: str = "bottle"                 # stage 1 and 2, None for ALLCAT
    usd_cache: str = ""                        # the converted corpus
    corpus_root: str = ""                      # meshdatav3_scaled, for the viewer
    visual_feat_root: str = ""                 # meshdatav3_pc_feat, the 64-dim obs block
    robot_urdf_src: str = ""                   # local fr3_xhand_dex4d.urdf, viewer reads it
    curriculum: bool = True
    stage2_start_iteration: int = 15000        # absent from every FR3 yaml
    random_time: bool = True                   # yaml :12, one-shot phase stagger
    seed: int = 0                              # the gym launcher passes --seed=0

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
        """cfg/fr3_xhand_ap2ap_stage_3.yaml. The reset pose is NOT changed, both
        stages use SimToolReal's."""
        import copy
        c = copy.deepcopy(self)
        c.object_cls = None
        # yaml says 12 against a 1/60 step. On the 1/120 step it is 24 for the
        # same 5 Hz.
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
