#!/usr/bin/env bash
set -euo pipefail

cd /home/najla/dev/najla-msc
mkdir -p bikeshare/results/semantic_enhancement/mlp
rm -rf bikeshare/scripts/semantic_enhancement/mlp/__pycache__

LOG=/home/najla/dev/najla-msc/bikeshare/results/semantic_enhancement/mlp/mlp_smoke_test_gpu.log
PIDFILE=/home/najla/dev/najla-msc/bikeshare/results/semantic_enhancement/mlp/mlp_smoke_test_gpu.pid

: > "$LOG"
{
  echo "Started MLP smoke test at $(date -Is)"
  echo "Working directory: /home/najla/dev/najla-msc"
  echo "Command: CUDA_VISIBLE_DEVICES=0 MLP_WIKI_MODE=pca32 PYTHONUNBUFFERED=1 .venv/bin/python -u bikeshare/scripts/semantic_enhancement/mlp/mlp_smoke_test.py"
} >> "$LOG"

CUDA_VISIBLE_DEVICES=0 \
MLP_WIKI_MODE=pca32 \
PYTHONUNBUFFERED=1 \
nohup .venv/bin/python -u bikeshare/scripts/semantic_enhancement/mlp/mlp_smoke_test.py >> "$LOG" 2>&1 &

pid=$!
echo "$pid" > "$PIDFILE"
echo "PID=$pid"
echo "LOG=$LOG"
echo "PIDFILE=$PIDFILE"