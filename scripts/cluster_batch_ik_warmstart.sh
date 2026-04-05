#!/bin/bash
# cluster_batch_ik_warmstart.sh — CPU warm-started IK on Green dataset
#
# Multiprocessing across trials (8 workers), warm-start within trials.
# ~63fps per worker × 8 = ~500fps total. 2.5M active frames ≈ 1.5 hours.
#
# Usage:
#   bash scripts/cluster_batch_ik_warmstart.sh [--dry-run]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"

GREEN_DIR="/groups/johnson/johnsonlab/virtual_rodent/green"
MODEL="/groups/johnson/johnsonlab/virtual_rodent/body_model/rodent_red_mj_dev_fitted.mjb"
OUTPUT="$GREEN_DIR/qpos_warmstart.csv"
LOG="$GREEN_DIR/batch_ik_warmstart.log"
WORKERS=8
DRY_RUN=false
RUN_MODE=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --dry-run)  DRY_RUN=true; shift ;;
        --run)      RUN_MODE=true; shift ;;
        *) echo "Unknown: $1"; exit 1 ;;
    esac
done

if $RUN_MODE; then
    echo "=== Warm-Start Batch IK ==="
    echo "Host: $(hostname), Date: $(date)"
    source ~/miniconda3/bin/activate && conda activate mjx
    pip install -e "$REPO_DIR" --quiet 2>&1 | tail -3

    python3 "$REPO_DIR/scripts/batch_ik_warmstart.py" \
        --green-dir "$GREEN_DIR" \
        --model "$MODEL" \
        --traj repaired_traj3d.bin \
        --metrics-csv "$GREEN_DIR/computed_metrics.csv" \
        --output "$OUTPUT" \
        --workers $WORKERS

    echo "=== Done ==="
    exit 0
fi

BSUB_CMD="bsub -W 4:00 -n $WORKERS -q local -P johnson \
    -J ik_warmstart \
    -o $LOG \
    bash $REPO_DIR/scripts/cluster_batch_ik_warmstart.sh --run"

echo "Warm-Start Batch IK (CPU, $WORKERS workers)"
echo "  Green:  $GREEN_DIR"
echo "  Model:  $MODEL"
echo "  Output: $OUTPUT"
echo "  Log:    $LOG"
echo ""

if $DRY_RUN; then
    echo "[dry-run] $BSUB_CMD"
else
    eval "$BSUB_CMD"
    echo "Monitor: bjobs -w | grep ik_warm"
    echo "Log:     $LOG"
fi
