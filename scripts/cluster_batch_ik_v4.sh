#!/bin/bash
# cluster_batch_ik_v4.sh — Batch IK with per-animal v4 fitted models
#
# Uses the correct fitted body model for each rat. ~3.5 hours on 8 cores.
#
# Usage:
#   bash scripts/cluster_batch_ik_v4.sh [--dry-run]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"

GREEN_DIR="/groups/johnson/johnsonlab/virtual_rodent/green"
MODELS_DIR="/groups/johnson/johnsonlab/virtual_rodent/body_model/green_fits_v4"
OUTPUT="$GREEN_DIR/qpos_v4.csv"
LOG="$GREEN_DIR/batch_ik_v4.log"

DRY_RUN=false
RUN_MODE=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --dry-run) DRY_RUN=true; shift ;;
        --run)     RUN_MODE=true; shift ;;
        *) echo "Unknown: $1"; exit 1 ;;
    esac
done

if $RUN_MODE; then
    echo "=== Batch IK v4 (per-animal models) ==="
    echo "Host: $(hostname), Date: $(date)"

    source ~/miniconda3/bin/activate && conda activate mjx
    pip install -e "$REPO_DIR" duckdb --quiet 2>&1 | tail -3 || true

    python3 "$REPO_DIR/scripts/batch_ik_v4.py" \
        --green-dir "$GREEN_DIR" \
        --models-dir "$MODELS_DIR" \
        --output "$OUTPUT" \
        --traj repaired_traj3d.bin \
        --workers 8

    echo "=== Done ==="
    exit 0
fi

BSUB_CMD="bsub -W 6:00 -n 8 -q local -P johnson \
    -R \"rusage[mem=16000]\" \
    -J ik_v4 \
    -o $LOG \
    bash $REPO_DIR/scripts/cluster_batch_ik_v4.sh --run"

echo "Batch IK v4 (per-animal models)"
echo "  Green:   $GREEN_DIR"
echo "  Models:  $MODELS_DIR"
echo "  Output:  $OUTPUT"
echo "  Log:     $LOG"
echo ""

if $DRY_RUN; then
    echo "[dry-run] $BSUB_CMD"
else
    eval "$BSUB_CMD"
    echo "Monitor: bjobs -w | grep ik_v4"
    echo "Log:     tail -f $LOG"
fi
