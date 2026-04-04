#!/bin/bash
# cluster_test_gpu_ik.sh — Validate MJX IK solver on A100 GPU
#
# Runs compare_ik.py on tiny_project data to verify:
#   1. GPU compilation time (~30s expected vs 20min on CPU)
#   2. Scan mode (full JIT vmap+scan) works on GPU
#   3. Residuals match CPU baseline (<2mm difference)
#   4. Throughput (frames/sec) on GPU
#
# Usage:
#   # From login node:
#   bash scripts/cluster_test_gpu_ik.sh [--dry-run]
#
#   # Or submit directly:
#   bsub -W 1:00 -n 8 -gpu "num=1" -q gpu_a100 -P johnson \
#     -o /groups/johnson/johnsonlab/virtual_rodent/gpu_ik_test.log \
#     bash scripts/cluster_test_gpu_ik.sh --run

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"

DATA_DIR="/groups/johnson/johnsonlab/virtual_rodent/tiny_project"
MODEL="/groups/johnson/johnsonlab/virtual_rodent/body_model/rodent_red_mj_dev_fitted.mjb"
CONDA_ENV="mjx"
DRY_RUN=false
RUN_MODE=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --dry-run)  DRY_RUN=true; shift ;;
        --run)      RUN_MODE=true; shift ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

if $RUN_MODE; then
    # Running inside the job — activate env and run
    echo "=== MJX GPU IK Validation ==="
    echo "Host: $(hostname)"
    echo "Date: $(date)"

    source ~/miniconda3/bin/activate
    conda activate $CONDA_ENV

    echo ""
    echo "--- Python/JAX info ---"
    python3 -c "
import jax
print(f'JAX {jax.__version__}')
print(f'Devices: {jax.devices()}')
gpu = [d for d in jax.devices() if d.platform == 'gpu']
print(f'GPU count: {len(gpu)}')
if gpu: print(f'GPU: {gpu[0]}')
"

    echo ""
    echo "--- Installing adjustabodies ---"
    pip install -e "$REPO_DIR" --quiet 2>&1 | tail -3

    echo ""
    echo "--- Test 1: Quick validation (20 frames, 500 iters) ---"
    python3 "$REPO_DIR/scripts/compare_ik.py" \
        --data-dir "$DATA_DIR" \
        --model "$MODEL" \
        --max-frames 20 \
        --ik-iters 500 \
        --lr 0.01

    echo ""
    echo "--- Test 2: Full tiny_project (all frames, 1000 iters) ---"
    python3 "$REPO_DIR/scripts/compare_ik.py" \
        --data-dir "$DATA_DIR" \
        --model "$MODEL" \
        --ik-iters 1000 \
        --lr 0.01 \
        --reference-csv "$DATA_DIR/qpos_export.csv"

    echo ""
    echo "=== Done ==="
    exit 0
fi

# Submit mode — create and submit the job
LOG="/groups/johnson/johnsonlab/virtual_rodent/gpu_ik_test.log"

BSUB_CMD="bsub -W 1:00 -n 8 -gpu \"num=1\" -q gpu_a100 -P johnson \
    -J gpu_ik_test \
    -o $LOG \
    bash $REPO_DIR/scripts/cluster_test_gpu_ik.sh --run"

echo "MJX GPU IK Validation Test"
echo "  Data:  $DATA_DIR"
echo "  Model: $MODEL"
echo "  Log:   $LOG"
echo ""

if $DRY_RUN; then
    echo "[dry-run] $BSUB_CMD"
else
    eval "$BSUB_CMD"
    echo ""
    echo "Monitor: bjobs -w | grep gpu_ik"
    echo "Log:     tail -f $LOG"
fi
