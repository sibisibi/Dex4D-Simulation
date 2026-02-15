python train.py \
--task=XArm6LeapHandAP2APVision \
--algo=dagger \
--seed=0 \
--rl_device=cuda:0 \
--sim_device=cuda:0 \
--logdir=<YOUR_LOG_DIR> \
--expert_model_dir=<CKPT_NAME_FROM_STAGE_3>.pt \
--headless \
--max_iterations=25000 \
# --test
