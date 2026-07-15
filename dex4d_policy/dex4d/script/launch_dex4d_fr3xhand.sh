#!/bin/bash
# Dex4D teacher PPO stage 1&2 with FR3 + LeFranX bracket + XHand,
# wandb draftrec/play as dex4d-fr3xhand_<ts>.
# 018 re-run: pose-viewer media logged from iteration 0, GPU as $1 (yj-3 GPU 1 default).
set -e
cd "$(dirname "$0")/.."

source /home/nas_main/sibeenkim/anaconda3/etc/profile.d/conda.sh
conda activate dex4d-sim
export LD_LIBRARY_PATH=/home/nas_main/sibeenkim/anaconda3/envs/dex4d-sim/lib:$LD_LIBRARY_PATH
export CUDA_VISIBLE_DEVICES=${1:-1}

cp cfg/fr3_xhand_ap2ap_stage_1_2.yaml cfg/fr3_xhand_ap2ap.yaml
cp cfg/ppo/config_stage_1_2.yaml cfg/ppo/config.yaml

ts=$(date +%Y%m%d_%H%M%S)
LOG=logs/fr3_xhand_ap2ap/ppo/dex4d-fr3xhand_$ts
mkdir -p "$LOG"

python train.py --task=FR3XHandAP2AP --algo=ppo --seed=0 \
  --rl_device=cuda:0 --sim_device=cuda:0 \
  --logdir="$LOG" --headless --max_iterations=25000 \
  --wandb_project play --wandb_entity draftrec \
  --capture_viewer --capture_viewer_url_check error \
  2>&1 | tee "$LOG/train.log"
