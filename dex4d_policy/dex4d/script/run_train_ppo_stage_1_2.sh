#!/bin/bash
# Usage: run_train_ppo_stage_1_2.sh <GPU> <STAGE1_ITERS> <STAGE2_ITERS>
# Flip at STAGE1_ITERS, total = STAGE1_ITERS + STAGE2_ITERS.
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
S1=$2
S2=$3
TOTAL=$((S1 + S2))

ts=$(date +%Y%m%d_%H%M%S)
LOG=logs/fr3_xhand_ap2ap/ppo/fr3xhand-s1-${S1}-s2-${S2}_$ts
mkdir -p "$LOG"

# python train.py \
# --task=XArm6LeapHandAP2AP \
# --algo=ppo \
# --seed=0 \
# --rl_device=cuda:0 \
# --sim_device=cuda:0 \
# --logdir=<YOUR_LOG_DIR> \
# --headless \
# --max_iterations=25000 \
# #--test

python train.py \
--task=FR3XHandAP2AP \
--algo=ppo \
--seed=0 \
--rl_device=cuda:0 \
--sim_device=cuda:0 \
--logdir="$LOG" \
--headless \
--max_iterations=$TOTAL \
--cfg_env cfg/fr3_xhand_ap2ap_stage_1_2.yaml \
--cfg_train cfg/ppo/config_stage_1_2.yaml \
--stage2_start_iteration=$S1 \
--capture_viewer --capture_viewer_url_check error \
2>&1 | tee "$LOG/train.log"
