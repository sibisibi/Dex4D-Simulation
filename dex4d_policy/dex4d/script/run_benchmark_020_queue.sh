#!/bin/bash
# 020 benchmark worker queue. Runs one shard of units sequentially on one GPU.
# Shard files hold "SET KEY" lines; a failed unit is recorded in failed.txt and
# the queue continues (the failure stays loud in the unit's train.log).
# Usage: run_benchmark_020_queue.sh <GPU> <SHARD_FILE>
set -e
cd "$(dirname "$0")/.."

GPU=$1
SHARD=$2
test -f "$SHARD"

CONDA_ROOT=${CONDA_ROOT:-$(ls -d /home/nas*/sibeenkim/anaconda3 2>/dev/null | head -1)}
source $CONDA_ROOT/etc/profile.d/conda.sh
conda activate dex4d-sim
export LD_LIBRARY_PATH=$CONDA_ROOT/envs/dex4d-sim/lib:$LD_LIBRARY_PATH
export PATH=$CONDA_ROOT/envs/dex4d-sim/bin:$PATH
export CPATH=$CONDA_ROOT/envs/dex4d-sim/include
export CUDA_VISIBLE_DEVICES=$GPU

RUNS=/home/nas5/sibeenkim/work/_020-diverse-eval/runs_dex4d

while read -r SET KEY; do
  SAFE=${KEY//\//__}
  SAFE=${SAFE//@/_s}
  OUT=$RUNS/$SET/$SAFE
  if [ -f "$OUT/benchmark_eval.json" ]; then
    echo "[skip] $SET $KEY"
    continue
  fi
  echo "[run ] $SET $KEY -> $OUT"
  if ! python eval_benchmark_020.py --key "$KEY" --set "$SET" --out_dir "$OUT"; then
    echo "$SET $KEY" >> "$RUNS/failed.txt"
    echo "[FAIL] $SET $KEY (recorded in $RUNS/failed.txt)"
  fi
done < "$SHARD"
echo "[done] shard $SHARD"
