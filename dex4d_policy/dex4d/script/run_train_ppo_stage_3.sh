python train.py \
--task=XArm6LeapHandAP2AP \
--algo=ppo \
--seed=0 \
--rl_device=cuda:0 \
--sim_device=cuda:0 \
--logdir=<YOUR_LOG_DIR> \
--headless \
--max_iterations=50000 \
--model_dir=<CKPT_NAME_FROM_STAGE_1_2>.pt \
#--test