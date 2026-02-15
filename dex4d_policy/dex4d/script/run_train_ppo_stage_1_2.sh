python train.py \
--task=XArm6LeapHandAP2AP \
--algo=ppo \
--seed=0 \
--rl_device=cuda:0 \
--sim_device=cuda:0 \
--logdir=<YOUR_LOG_DIR> \
--headless \
--max_iterations=25000 \
#--test