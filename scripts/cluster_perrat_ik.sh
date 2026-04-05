#!/bin/bash
# cluster_perrat_ik.sh — Per-rat IK evaluation on Green data (A100 GPU)
#
# Usage:
#   bash scripts/cluster_perrat_ik.sh [--dry-run]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"

GREEN_DIR="/groups/johnson/johnsonlab/virtual_rodent/green"
MODEL="/groups/johnson/johnsonlab/virtual_rodent/body_model/rodent_red_mj_dev_fitted.mjb"
LOG="$GREEN_DIR/perrat_ik.log"
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
    echo "=== Per-Rat IK Evaluation ==="
    source ~/miniconda3/bin/activate && conda activate mjx
    pip install -e "$REPO_DIR" duckdb --quiet 2>&1 | tail -3

    python3 "$REPO_DIR/scripts/perrat_ik_green.py" \
        --green-dir "$GREEN_DIR" \
        --green-db "$GREEN_DIR/green.duckdb" \
        --model "$MODEL" \
        --traj repaired_traj3d.bin \
        --frames-per-rat 500 \
        --ik-iters 1000 \
        --lr 0.01

    echo "=== Done ==="
    exit 0
fi

BSUB_CMD="bsub -W 1:00 -n 8 -gpu \"num=1\" -q gpu_a100 -P johnson \
    -J perrat_ik \
    -o $LOG \
    bash $REPO_DIR/scripts/cluster_perrat_ik.sh --run"

echo "Per-Rat IK Evaluation (MJX GPU)"
echo "  Green: $GREEN_DIR"
echo "  Model: $MODEL"
echo "  Log:   $LOG"
echo ""

if $DRY_RUN; then
    echo "[dry-run] $BSUB_CMD"
else
    eval "$BSUB_CMD"
    echo "Monitor: bjobs -w | grep perrat"
    echo "Log:     tail -f $LOG"
fi
