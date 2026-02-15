MODEL_DIR=<CKPT_NAME>.pt
NUM_ENVS=50
VIS_ENV_ID=0


python train.py \
--task=XArm6LeapHandAP2APVision \
--algo=dagger \
--seed=0 \
--rl_device=cuda:0 \
--sim_device=cuda:0 \
--model_dir=$MODEL_DIR \
--test \
--num_envs=$NUM_ENVS \
--vis_env_id=$VIS_ENV_ID