import os.path as osp

from utils.util import copy_files

def process_ppo(args, env, cfg_train, logdir):
    from algorithms.rl.ppo import PPO, ActorCritic, ActorCriticPointNet
    learn_cfg = cfg_train["learn"]
    #is_testing = learn_cfg["test"]
    is_testing = args.test
    is_vision = args.vision
    # is_testing = True
    # Override resume and testing flags if they are passed as parameters.
    if args.model_dir != "":
        #is_testing = True
        chkpt_path = args.model_dir

    # logdir = logdir + "_seed{}".format(env.task.cfg["seed"])
    """Set up the PPO system for training or inferencing."""
    ppo = PPO(vec_env=env,
              actor_critic_class=eval(learn_cfg.get("actor_critic_class", "ActorCritic")),
              num_transitions_per_env=learn_cfg["nsteps"],
              num_learning_epochs=learn_cfg["noptepochs"],
              num_mini_batches=learn_cfg["nminibatches"],
              clip_param=learn_cfg["cliprange"],
              gamma=learn_cfg["gamma"],
              lam=learn_cfg["lam"],
              init_noise_std=learn_cfg.get("init_noise_std", 0.3),
              value_loss_coef=learn_cfg.get("value_loss_coef", 2.0),
              entropy_coef=learn_cfg["ent_coef"],
              learning_rate=learn_cfg["optim_stepsize"],
              max_grad_norm=learn_cfg.get("max_grad_norm", 2.0),
              use_clipped_value_loss=learn_cfg.get("use_clipped_value_loss", False),
              schedule=learn_cfg.get("schedule", "fixed"),
              desired_kl=learn_cfg.get("desired_kl", None),
              model_cfg=cfg_train["policy"],
              device=env.rl_device,
              sampler=learn_cfg.get("sampler", 'sequential'),
              log_dir=logdir,
              is_testing=is_testing,
              print_log=learn_cfg["print_log"],
              apply_reset=False,
              asymmetric=env.task.asymmetric_obs,
              is_vision=is_vision,
              wandb_project=args.wandb_project,
              wandb_entity=(args.wandb_entity or None),
              wandb_name=(args.wandb_name or None),
              )
    if not is_testing:
        copy_files(args, logdir)

    if (not is_testing) and args.capture_viewer:
        from utils.pose_viewer import Dex4DPoseViewer
        ppo.pose_viewer = Dex4DPoseViewer(
            env.task,
            output_dir=osp.join(logdir, "interactive_viewer"),
            capture_len=args.capture_viewer_len,
            capture_interval=args.capture_viewer_interval,
            env_id=args.capture_viewer_env_id,
            wandb_key=args.capture_viewer_wandb_key,
            robot_raw_base=args.capture_viewer_raw_base,
            url_check=args.capture_viewer_url_check,
        )

    if is_testing and args.model_dir != "":
        print("Loading model from {}".format(chkpt_path))
        ppo.test(chkpt_path)
    elif args.model_dir != "":
        print("Loading model from {}".format(chkpt_path))
        ppo.load(chkpt_path)

    return ppo

def process_dagger(args, env, cfg_train, logdir):
    from algorithms.rl.ppo import ActorCritic, ActorCriticPointNet
    from algorithms.rl.dagger import DAGGER, Actor, ActorPointNet, ActorPointNetTransformer
    learn_cfg = cfg_train["learn"]
    #is_testing = learn_cfg["test"]
    is_testing = args.test
    is_vision = args.vision
    # is_testing = True
    # Override resume and testing flags if they are passed as parameters.
    if args.model_dir != "":
        #is_testing = True
        chkpt_path = args.model_dir
    expert_chkpt_path = ""
    if args.expert_model_dir != "":
        #is_testing = True
        expert_chkpt_path = args.expert_model_dir

    # logdir = logdir + "_seed{}".format(env.task.cfg["seed"])
    """Set up the DAgger system for training or inferencing."""
    dagger = DAGGER(vec_env=env,
              actor_class=eval(learn_cfg.get("actor_class", "Actor")),
              actor_critic_class=eval(learn_cfg.get("actor_critic_class", "ActorCritic")),
              num_transitions_per_env=learn_cfg["nsteps"],
              num_learning_epochs=learn_cfg["noptepochs"],
              num_mini_batches=learn_cfg["nminibatches"],
              buffer_size=learn_cfg["buffer_size"],
              init_noise_std=learn_cfg.get("init_noise_std", 0.3),
              learning_rate=learn_cfg.get("learning_rate", 1e-3),
              schedule=learn_cfg.get("schedule", "fixed"),
              model_cfg=cfg_train["policy"],
              device=env.rl_device,
              sampler=learn_cfg.get("sampler", 'sequential'),
              log_dir=logdir,
              is_testing=is_testing,
              print_log=learn_cfg["print_log"],
              apply_reset=False,
              asymmetric=env.task.asymmetric_obs,
              expert_chkpt_path = expert_chkpt_path,
              is_vision = is_vision
              )
    if not is_testing:
        copy_files(args, logdir)

    if (not is_testing) and args.capture_viewer:
        from utils.pose_viewer import Dex4DPoseViewer
        dagger.pose_viewer = Dex4DPoseViewer(
            env.task,
            output_dir=osp.join(logdir, "interactive_viewer"),
            capture_len=args.capture_viewer_len,
            capture_interval=args.capture_viewer_interval,
            env_id=args.capture_viewer_env_id,
            wandb_key=args.capture_viewer_wandb_key,
            robot_raw_base=args.capture_viewer_raw_base,
            url_check=args.capture_viewer_url_check,
        )

    if is_testing and args.model_dir != "":
        print("Loading model from {}".format(chkpt_path))
        dagger.test(chkpt_path)
    elif args.model_dir != "":
        print("Loading model from {}".format(chkpt_path))
        dagger.load(chkpt_path)

    return dagger
