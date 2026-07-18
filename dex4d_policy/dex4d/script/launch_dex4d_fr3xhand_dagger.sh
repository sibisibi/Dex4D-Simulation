#!/bin/bash
# Portable FR3+bracket+XHand DAgger distillation launcher (host-agnostic conda).
# Distills the stage-3 full-state teacher into the partial-obs PointNet-transformer
# student on FR3XHandAP2AP. The staged env cfg is the stage-3 yaml plus the upstream
# vision deltas, which equal the post-curriculum-flip env the teacher converged in.
# Deltas. asymmetric obs on, too_far reset off, stable ratio 0.2, dofSpeedScale 1.5,
# 3200 envs, curriculum off. Control stays 5 Hz to match the teacher.
# The student starts fresh at iteration 0, so MAX_ITER is the total budget.
# wandb. dagger.py hardcodes project Dex4D-DAGGER and reads no entity kwarg, so the
# entity comes from WANDB_ENTITY. The --wandb_project/--wandb_entity flags are
# consumed by the PPO path only today, kept here so the run is correct if dagger.py
# ever gets the same wandb kwargs as ppo.py.
# No --capture_viewer. The pose-viewer hook lives in the PPO rollout loop only.
# Usage: launch_dex4d_fr3xhand_dagger.sh <GPU> <TEACHER_CKPT_ABS_PATH> [MAX_ITER=25000] [LABEL=g57d5g51]
set -e
cd "$(dirname "$0")/.."

GPU=$1
TEACHER=$2
MAX_ITER=${3:-25000}
LABEL=${4:-g57d5g51}
test -f "$TEACHER"

CONDA_ROOT=${CONDA_ROOT:-$(cd ~ && ls -d /home/nas*/sibeenkim/anaconda3 2>/dev/null | head -1)}
source $CONDA_ROOT/etc/profile.d/conda.sh
conda activate dex4d-sim-dagger
export LD_LIBRARY_PATH=$CONDA_ROOT/envs/dex4d-sim-dagger/lib:$LD_LIBRARY_PATH
export PATH=$CONDA_ROOT/envs/dex4d-sim-dagger/bin:$PATH
export CPATH=$CONDA_ROOT/envs/dex4d-sim-dagger/include
export CUDA_VISIBLE_DEVICES=$GPU
export WANDB_ENTITY=draftrec

# The deltas live in the committed cfg/fr3_xhand_ap2ap_dagger.yaml (stage-3 yaml
# with the six vision deltas applied once). Passed by direct path below, never
# staged into a shared active file, so concurrent launches are safe. Asserts only.
grep -q '^  numEnvs: 3200$' cfg/fr3_xhand_ap2ap_dagger.yaml
grep -q '^  too_far_reset_threshold: 1000000.0$' cfg/fr3_xhand_ap2ap_dagger.yaml
grep -q '^  goal_reset_stable_ratio: 0.2$' cfg/fr3_xhand_ap2ap_dagger.yaml
grep -q '^  dofSpeedScale: 1.5$' cfg/fr3_xhand_ap2ap_dagger.yaml
grep -q '^  asymmetric_observations: True' cfg/fr3_xhand_ap2ap_dagger.yaml
grep -q '^  curriculum: False$' cfg/fr3_xhand_ap2ap_dagger.yaml

# Pre-flight guard. ppo.py line 84 injects model_cfg['kp_start'] from the task,
# 204 for FR3, but DAGGER.__init__ does not, so the expert ActorCriticPointNet
# falls back to the stock offset 197 and silently mis-slices FR3 keypoints.
# Refuse to launch until dagger.py carries the same one-line injection.
grep -q "kp_start" algorithms/rl/dagger/dagger.py || {
  echo "FATAL, dagger.py lacks the model_cfg['kp_start'] injection, the FR3 expert would mis-slice keypoints at offset 197 instead of 204. Mirror ppo.py line 84 into DAGGER.__init__ next to the num_robot_dofs line, then relaunch." >&2
  exit 1
}

ts=$(date +%Y%m%d_%H%M%S)
LOG=logs/fr3_xhand_ap2ap/dagger/dex4d-fr3xhand-dagger-from-${LABEL}_$ts
mkdir -p "$LOG"

python train.py --task=FR3XHandAP2AP --algo=dagger --seed=0 \
  --rl_device=cuda:0 --sim_device=cuda:0 \
  --logdir="$LOG" --headless --max_iterations=$MAX_ITER \
  --cfg_env cfg/fr3_xhand_ap2ap_dagger.yaml --cfg_train cfg/dagger/config.yaml \
  --expert_model_dir="$TEACHER" \
  --wandb_project play --wandb_entity draftrec \
  2>&1 | tee "$LOG/train.log"
