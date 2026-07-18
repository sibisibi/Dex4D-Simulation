#!/bin/bash
# Portable FR3+bracket+XHand stage-3 launcher (host-agnostic conda + gymtorch CPATH).
# Warm-starts stage 3 (ALLCAT, 5 Hz) from a stage-1-2 final checkpoint; iteration
# resumes from the checkpoint filename, so MAX_ITER is the ABSOLUTE target.
# Usage: launch_dex4d_fr3xhand_stage3_portable.sh <GPU> <CKPT_ABS_PATH> <MAX_ITER>
set -e
cd "$(dirname "$0")/.."

GPU=$1
CKPT=$2
MAX_ITER=$3
test -f "$CKPT"

CONDA_ROOT=${CONDA_ROOT:-$(cd ~ && ls -d /home/nas*/sibeenkim/anaconda3 2>/dev/null | head -1)}
source $CONDA_ROOT/etc/profile.d/conda.sh
conda activate dex4d-sim
export LD_LIBRARY_PATH=$CONDA_ROOT/envs/dex4d-sim/lib:$LD_LIBRARY_PATH
export PATH=$CONDA_ROOT/envs/dex4d-sim/bin:$PATH
export CPATH=$CONDA_ROOT/envs/dex4d-sim/include
export CUDA_VISIBLE_DEVICES=$GPU

cp cfg/fr3_xhand_ap2ap_stage_3.yaml cfg/fr3_xhand_ap2ap.yaml
cp cfg/ppo/config_stage_3.yaml cfg/ppo/config.yaml

ts=$(date +%Y%m%d_%H%M%S)
LOG=logs/fr3_xhand_ap2ap/ppo/dex4d-fr3xhand-stage3-from-zl0hpl7g_$ts
mkdir -p "$LOG"

python train.py --task=FR3XHandAP2AP --algo=ppo --seed=0 \
  --rl_device=cuda:0 --sim_device=cuda:0 \
  --logdir="$LOG" --headless --max_iterations=$MAX_ITER \
  --model_dir="$CKPT" \
  --wandb_project play --wandb_entity draftrec \
  --capture_viewer --capture_viewer_url_check error \
  2>&1 | tee "$LOG/train.log"
