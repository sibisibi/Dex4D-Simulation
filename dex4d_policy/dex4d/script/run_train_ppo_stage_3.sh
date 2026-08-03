#!/bin/bash
# Warm-starts stage 3 from a stage-1-2 final checkpoint; iteration resumes from
# the checkpoint filename, so MAX_ITER is the ABSOLUTE target. RUN_NAME carries
# the full lineage and becomes the logdir basename and the wandb run name.
# Usage: run_train_ppo_stage_3.sh <GPU> <CKPT_ABS_PATH> <MAX_ITER> <RUN_NAME>
set -e
cd "$(dirname "$0")/.."

GPU=$1
CONDA_ROOT=${CONDA_ROOT:-$(cd ~ && ls -d /home/nas*/sibeenkim/anaconda3 2>/dev/null | head -1)}
source $CONDA_ROOT/etc/profile.d/conda.sh
conda activate dex4d-sim
export LD_LIBRARY_PATH=$CONDA_ROOT/envs/dex4d-sim/lib:$LD_LIBRARY_PATH
export PATH=$CONDA_ROOT/envs/dex4d-sim/bin:$PATH
export CPATH=$CONDA_ROOT/envs/dex4d-sim/include
export CUDA_VISIBLE_DEVICES=$GPU
CKPT=$2
MAX_ITER=$3
NAME=$4
test -n "$NAME" || { echo "FATAL, RUN_NAME (arg 4) is required" >&2; exit 1; }
test -f "$CKPT"

ts=$(date +%Y%m%d_%H%M%S)
LOG=logs/fr3_xhand_ap2ap/ppo/${NAME}_$ts
mkdir -p "$LOG"

# python train.py \
# --task=XArm6LeapHandAP2AP \
# --algo=ppo \
# --seed=0 \
# --rl_device=cuda:0 \
# --sim_device=cuda:0 \
# --logdir=<YOUR_LOG_DIR> \
# --headless \
# --max_iterations=50000 \
# --model_dir=<CKPT_NAME_FROM_STAGE_1_2>.pt \
# #--test

python train.py \
--task=FR3XHandAP2AP \
--algo=ppo \
--seed=0 \
--rl_device=cuda:0 \
--sim_device=cuda:0 \
--logdir="$LOG" \
--headless \
--max_iterations=$MAX_ITER \
--cfg_env cfg/fr3_xhand_ap2ap_stage_3.yaml \
--cfg_train cfg/ppo/config_stage_3.yaml \
--model_dir="$CKPT" \
--wandb_name "${NAME}_$ts" \
--capture_viewer --capture_viewer_url_check error \
2>&1 | tee "$LOG/train.log"
