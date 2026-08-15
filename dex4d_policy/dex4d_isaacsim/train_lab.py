"""Train Dex4D on IsaacLab with Dex4D's own PPO."""
import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--port", required=True)
parser.add_argument("--repo", required=True)
parser.add_argument("--cache", required=True)
parser.add_argument("--keypoints", required=True)
parser.add_argument("--robot_urdf", required=True)
parser.add_argument("--arm_gain_row", default="a3")
parser.add_argument("--num_envs", type=int, default=256)
parser.add_argument("--max_iterations", type=int, default=25000)
parser.add_argument("--stage2_start_iteration", type=int, default=15000)
parser.add_argument("--logdir", required=True)
parser.add_argument("--wandb_project", default="play")
parser.add_argument("--wandb_entity", default="draftrec")
parser.add_argument("--wandb_name", default="")
parser.add_argument("--corpus_root", default="")
parser.add_argument("--visual_feat_root", required=True)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--stage3", action="store_true")
parser.add_argument("--model_dir", default="")
parser.add_argument("--capture_viewer", action="store_true")
parser.add_argument("--capture_viewer_len", type=int, default=600)
parser.add_argument("--capture_viewer_interval", type=int, default=1000)
parser.add_argument("--capture_viewer_url_check", default="error")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True

simulation_app = AppLauncher(args).app

import os  # noqa: E402
import sys  # noqa: E402
import tempfile  # noqa: E402
from pathlib import Path  # noqa: E402

import torch  # noqa: E402
import yaml  # noqa: E402
from isaaclab.envs import DirectRLEnvCfg  # noqa: E402
from isaaclab.scene import InteractiveSceneCfg  # noqa: E402
from isaaclab.sim import SimulationCfg  # noqa: E402
import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.utils import configclass  # noqa: E402

from isaacsimenvs.tasks.simtoolreal.utils import scene_utils as ST  # noqa: E402


def set_seed(seed):
    """Seed python, numpy and torch. Non-deterministic cudnn."""
    import random

    import numpy as np

    print(f"Setting seed: {seed}", flush=True)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False


def main():
    set_seed(args.seed)
    sys.path.insert(0, args.port)
    sys.path.insert(0, str(Path(args.repo) / "dex4d_policy/dex4d"))
    from env import Dex4DEnv  # noqa: E402
    from env_cfg import Dex4DEnvCfg  # noqa: E402
    from vec_adapter import Dex4DVecAdapter  # noqa: E402
    # The RL stack imports gymapi at module level, which cannot load here.
    import isaacgym_stub  # noqa: E402
    isaacgym_stub.install()
    from algorithms.rl.ppo import PPO, ActorCritic, ActorCriticPointNet  # noqa: E402,F401

    work = Path(tempfile.mkdtemp(prefix="d4train_"))
    usd = ST._convert_urdf_to_usd(args.robot_urdf, work, fix_base=True,
                                  self_collision=True,
                                  joint_drive=ST._robot_joint_drive_cfg())
    usd = ST._bake_usd(usd, work / "bake", "robot",
                       props=dict(disable_gravity=True,
                                  max_depenetration_velocity=1000.0,
                                  enabled_self_collisions=True,
                                  solver_position_iterations=8,
                                  solver_velocity_iterations=0),
                       apply_physx_articulation=True)

    d4 = Dex4DEnvCfg()
    d4.usd_cache, d4.robot_usd, d4.keypoint_npz = args.cache, usd, args.keypoints
    d4.arm_gain_row = args.arm_gain_row
    d4.num_envs = args.num_envs
    d4.stage2_start_iteration = args.stage2_start_iteration
    d4.corpus_root = args.corpus_root
    d4.visual_feat_root = args.visual_feat_root
    d4.robot_urdf_src = args.robot_urdf
    if args.stage3:
        d4 = d4.stage3()
    px = d4.physx

    @configclass
    class Cfg(DirectRLEnvCfg):
        # Keeps the simulator and object randomization on the policy's seed.
        seed = args.seed
        decimation = d4.control.control_frequency_inv
        episode_length_s = (d4.termination.episode_length
                            * d4.control.control_frequency_inv * d4.control.sim_dt)
        action_space = 19
        observation_space = 1051
        state_space = 0
        sim = SimulationCfg(
            dt=d4.control.sim_dt, render_interval=d4.control.control_frequency_inv,
            physx=sim_utils.PhysxCfg(
                solver_type=px.solver_type,
                max_position_iteration_count=px.num_position_iterations,
                max_velocity_iteration_count=px.num_velocity_iterations,
                bounce_threshold_velocity=px.bounce_threshold_velocity))
        scene = InteractiveSceneCfg(num_envs=args.num_envs,
                                    env_spacing=d4.env_spacing,
                                    replicate_physics=False)
        dex4d = d4

    print(f"\n=== building, {args.num_envs} envs, arm gain row {args.arm_gain_row}",
          flush=True)
    env = Dex4DEnv(Cfg())
    if args.capture_viewer:
        from pose_viewer import Dex4DPoseViewerWrapper  # noqa: E402
        env = Dex4DPoseViewerWrapper(
            env, output_dir=Path(args.logdir) / "interactive_viewer",
            capture_len=args.capture_viewer_len,
            capture_interval=args.capture_viewer_interval,
            wandb_key="interactive_viewer",
            url_check=args.capture_viewer_url_check)
    vec = Dex4DVecAdapter(env, rl_device=str(env.unwrapped.device),
                          clip_observations=d4.obs.clip_observations)
    print(f"  obs {vec.observation_space.shape}  state {vec.state_space.shape}  "
          f"act {vec.action_space.shape}  kp_start {vec.task.kp_start}", flush=True)

    ppo_yaml = "config_stage_3.yaml" if args.stage3 else "config_stage_1_2.yaml"
    cfg_train = yaml.safe_load(
        (Path(args.repo) / "dex4d_policy/dex4d/cfg/ppo" / ppo_yaml).read_text())
    learn = cfg_train["learn"]
    Path(args.logdir).mkdir(parents=True, exist_ok=True)

    runner = PPO(
        vec_env=vec,
        actor_critic_class=eval(learn.get("actor_critic_class", "ActorCritic")),
        num_transitions_per_env=learn["nsteps"],
        num_learning_epochs=learn["noptepochs"],
        num_mini_batches=learn["nminibatches"],
        clip_param=learn["cliprange"],
        gamma=learn["gamma"],
        lam=learn["lam"],
        init_noise_std=learn.get("init_noise_std", 0.3),
        value_loss_coef=learn.get("value_loss_coef", 2.0),
        entropy_coef=learn["ent_coef"],
        learning_rate=float(learn["optim_stepsize"]),
        max_grad_norm=learn.get("max_grad_norm", 2.0),
        use_clipped_value_loss=learn.get("use_clipped_value_loss", False),
        schedule=learn.get("schedule", "fixed"),
        desired_kl=learn.get("desired_kl", None),
        model_cfg=cfg_train["policy"],
        device=str(env.unwrapped.device),
        sampler=learn.get("sampler", "sequential"),
        log_dir=args.logdir,
        is_testing=False,
        print_log=learn["print_log"],
        apply_reset=False,
        asymmetric=d4.obs.asymmetric_observations,
        wandb_project=args.wandb_project,
        wandb_entity=args.wandb_entity,
        wandb_name=args.wandb_name or None,
    )
    if args.model_dir:
        runner.load(args.model_dir)
        print(f"  resumed at {runner.current_learning_iteration}", flush=True)
    print(f"\n=== PPO built, running to {args.max_iterations}", flush=True)
    runner.run(num_learning_iterations=args.max_iterations,
               log_interval=learn.get("save_interval", 100))

    simulation_app.close()
    os._exit(0)


if __name__ == "__main__":
    main()
