#!/usr/bin/env bash
set -uo pipefail

cd /home/najla/dev/najla-msc
mkdir -p bikeshare/results/semantic_enhancement/mlp
rm -rf bikeshare/scripts/semantic_enhancement/mlp/__pycache__

LOG=/home/najla/dev/najla-msc/bikeshare/results/semantic_enhancement/mlp/mlp_smoke_test_gpu_pca64.log
PIDFILE=/home/najla/dev/najla-msc/bikeshare/results/semantic_enhancement/mlp/mlp_smoke_test_gpu_pca64.pid

: > "$LOG"
echo "$$" > "$PIDFILE"
exec > >(tee -a "$LOG") 2>&1

echo "Started MLP smoke test pca64 at $(date -Is)"
echo "Working directory: /home/najla/dev/najla-msc"
echo "CUDA_VISIBLE_DEVICES=0"
echo "MLP_WIKI_MODE=pca64"
echo "PYTHONUNBUFFERED=1"
echo "Command: .venv/bin/python -u bikeshare/scripts/semantic_enhancement/mlp/mlp_smoke_test.py"

export CUDA_VISIBLE_DEVICES=0
export MLP_WIKI_MODE=pca64
export PYTHONUNBUFFERED=1

.venv/bin/python -u bikeshare/scripts/semantic_enhancement/mlp/mlp_smoke_test.py
status=$?

echo "Finished MLP smoke test pca64 at $(date -Is) with exit code $status"
exit "$status"