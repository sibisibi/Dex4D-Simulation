#!/bin/bash
# Dex4D FR3+bracket+XHand teacher PPO stage 3 (ALLCAT, 5 Hz), continues a stage 1-2 final
# checkpoint from iteration 25000 to 50000 (iteration parsed from the filename).
# Usage: launch_dex4d_fr3xhand_stage3.sh <GPU> <STAGE12_FINAL_CKPT>
set -e
cd "$(dirname "$0")/.."

CKPT=$2
test -f "$CKPT"

source /home/nas_main/sibeenkim/anaconda3/etc/profile.d/conda.sh
conda activate dex4d-sim
export LD_LIBRARY_PATH=/home/nas_main/sibeenkim/anaconda3/envs/dex4d-sim/lib:$LD_LIBRARY_PATH
export CUDA_VISIBLE_DEVICES=${1:-3}

cp cfg/fr3_xhand_ap2ap_stage_3.yaml cfg/fr3_xhand_ap2ap.yaml
cp cfg/ppo/config_stage_3.yaml cfg/ppo/config.yaml

ts=$(date +%Y%m%d_%H%M%S)
LOG=logs/fr3_xhand_ap2ap/ppo/dex4d-fr3xhand-stage3_$ts
mkdir -p "$LOG"

python train.py --task=FR3XHandAP2AP --algo=ppo --seed=0 \
  --rl_device=cuda:0 --sim_device=cuda:0 \
  --logdir="$LOG" --headless --max_iterations=50000 \
  --model_dir="$CKPT" \
  --wandb_project play --wandb_entity draftrec \
  --capture_viewer --capture_viewer_url_check error \
  2>&1 | tee "$LOG/train.log"
