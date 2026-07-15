#!/bin/bash
# Stock Dex4D teacher PPO stage 1&2 on GPU2, wandb draftrec/play as dex4d-default_<ts>
set -e
cd "$(dirname "$0")/.."

source /home/nas_main/sibeenkim/anaconda3/etc/profile.d/conda.sh
conda activate dex4d-sim
export LD_LIBRARY_PATH=/home/nas_main/sibeenkim/anaconda3/envs/dex4d-sim/lib:$LD_LIBRARY_PATH
export CUDA_VISIBLE_DEVICES=2

cp cfg/xarm6_leap_hand_ap2ap_stage_1_2.yaml cfg/xarm6_leap_hand_ap2ap.yaml
cp cfg/ppo/config_stage_1_2.yaml cfg/ppo/config.yaml

ts=$(date +%Y%m%d_%H%M%S)
LOG=logs/xarm6_leap_hand_ap2ap/ppo/dex4d-default_$ts
mkdir -p "$LOG"

python train.py --task=XArm6LeapHandAP2AP --algo=ppo --seed=0 \
  --rl_device=cuda:0 --sim_device=cuda:0 \
  --logdir="$LOG" --headless --max_iterations=25000 \
  --wandb_project play --wandb_entity draftrec \
  2>&1 | tee "$LOG/train.log"
