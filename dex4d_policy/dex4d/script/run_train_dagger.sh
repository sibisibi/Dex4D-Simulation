#!/bin/bash
# Distills the stage-3 teacher into the vision student.
# Usage: run_train_dagger.sh <GPU> <EXPERT_CKPT_ABS_PATH> <MAX_ITER> <RUN_NAME>
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
LOG=logs/fr3_xhand_ap2ap_vision/dagger/${NAME}_$ts
mkdir -p "$LOG"

# python train.py \
# --task=XArm6LeapHandAP2APVision \
# --algo=dagger \
# --seed=0 \
# --rl_device=cuda:0 \
# --sim_device=cuda:0 \
# --logdir=<YOUR_LOG_DIR> \
# --expert_model_dir=<CKPT_NAME_FROM_STAGE_3>.pt \
# --headless \
# --max_iterations=25000 \
# # --test

python train.py \
--task=FR3XHandAP2APVision \
--algo=dagger \
--seed=0 \
--rl_device=cuda:0 \
--sim_device=cuda:0 \
--logdir="$LOG" \
--headless \
--max_iterations=$MAX_ITER \
--cfg_env cfg/fr3_xhand_ap2ap_vision.yaml \
--cfg_train cfg/dagger/config.yaml \
--expert_model_dir="$CKPT" \
--capture_viewer --capture_viewer_url_check error \
2>&1 | tee "$LOG/train.log"
