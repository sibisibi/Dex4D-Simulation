#!/bin/bash
# Portable FR3+bracket+XHand stage-3 launcher (host-agnostic conda + gymtorch CPATH).
# Warm-starts stage 3 (ALLCAT, 5 Hz) from a stage-1-2 final checkpoint; iteration
# resumes from the checkpoint filename, so MAX_ITER is the ABSOLUTE target.
# Usage: launch_dex4d_fr3xhand_stage3_portable.sh <GPU> <CKPT_ABS_PATH> <MAX_ITER> <RUN_NAME>
# RUN_NAME carries the full lineage, e.g. dex4d-fr3xhand-s1-20000-s2-15000-s3-30000.
# It becomes the logdir basename (with a _timestamp suffix) and the wandb run name.
set -e
cd "$(dirname "$0")/.."

GPU=$1
CKPT=$2
MAX_ITER=$3
NAME=$4
test -n "$NAME" || { echo "FATAL, RUN_NAME (arg 4) is required, e.g. dex4d-fr3xhand-s1-20000-s2-15000-s3-30000" >&2; exit 1; }
test -f "$CKPT"

CONDA_ROOT=${CONDA_ROOT:-$(cd ~ && ls -d /home/nas*/sibeenkim/anaconda3 2>/dev/null | head -1)}
source $CONDA_ROOT/etc/profile.d/conda.sh
conda activate dex4d-sim
export LD_LIBRARY_PATH=$CONDA_ROOT/envs/dex4d-sim/lib:$LD_LIBRARY_PATH
export PATH=$CONDA_ROOT/envs/dex4d-sim/bin:$PATH
export CPATH=$CONDA_ROOT/envs/dex4d-sim/include
export CUDA_VISIBLE_DEVICES=$GPU

# cfg goes in by direct path (--cfg_env/--cfg_train), never by staging a shared
# active file. Concurrent launches of different stages on one clone are safe.

ts=$(date +%Y%m%d_%H%M%S)
LOG=logs/fr3_xhand_ap2ap/ppo/${NAME}_$ts
mkdir -p "$LOG"

python train.py --task=FR3XHandAP2AP --algo=ppo --seed=0 \
  --rl_device=cuda:0 --sim_device=cuda:0 \
  --logdir="$LOG" --headless --max_iterations=$MAX_ITER \
  --cfg_env cfg/fr3_xhand_ap2ap_stage_3.yaml --cfg_train cfg/ppo/config_stage_3.yaml \
  --model_dir="$CKPT" \
  --wandb_project play --wandb_entity draftrec --wandb_name "${NAME}_$ts" \
  --capture_viewer --capture_viewer_url_check error \
  2>&1 | tee "$LOG/train.log"
