#!/bin/bash
# cluster_fit_green.sh — Fit 6 body models for Green rats on A100 GPU
#
# 1 average model (500 frames × 5 rats) + 5 per-rat models (2500 frames each)
#
# Usage:
#   bash scripts/cluster_fit_green.sh [--dry-run]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"

GREEN_DIR="/groups/johnson/johnsonlab/virtual_rodent/green"
BASE_MODEL="/groups/johnson/johnsonlab/virtual_rodent/body_model/rodent_data_driven_limits.xml"
OUTPUT_DIR="/groups/johnson/johnsonlab/virtual_rodent/body_model/green_fits_v3"
LOG="$GREEN_DIR/fit_green_rats.log"
TARGET_WEIGHT=0.05
ONLY_RAT=""
DRY_RUN=false
RUN_MODE=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --dry-run)       DRY_RUN=true; shift ;;
        --run)           RUN_MODE=true; shift ;;
        --weight)        TARGET_WEIGHT="$2"; shift 2 ;;
        --only-rat)      ONLY_RAT="$2"; shift 2 ;;
        --output-dir)    OUTPUT_DIR="$2"; shift 2 ;;
        *) echo "Unknown: $1"; exit 1 ;;
    esac
done

if $RUN_MODE; then
    echo "=== Green Rat Body Model Fitting ==="
    echo "Host: $(hostname)"
    echo "Date: $(date)"

    source ~/miniconda3/bin/activate && conda activate mjx
    pip install -e "$REPO_DIR" duckdb --quiet 2>&1 | tail -3 || true

    python3 -c "import jax; print(f'JAX {jax.__version__}, {jax.devices()}')"

    EXTRA_ARGS=""
    if [[ -n "$ONLY_RAT" ]]; then
        EXTRA_ARGS="--only-rat $ONLY_RAT"
    fi

    python3 "$REPO_DIR/scripts/fit_green_rats.py" \
        --green-dir "$GREEN_DIR" \
        --green-db "$GREEN_DIR/green.duckdb" \
        --base-model "$BASE_MODEL" \
        --output-dir "$OUTPUT_DIR" \
        --traj repaired_traj3d.bin \
        --frames-per-rat 500 \
        --frames-average 100 \
        --n-rounds 6 \
        --m-iters 300 \
        --ik-iters 1000 \
        --target-weight "$TARGET_WEIGHT" \
        $EXTRA_ARGS

    echo "=== Done ==="
    exit 0
fi

BSUB_EXTRA=""
if [[ -n "$ONLY_RAT" ]]; then
    BSUB_EXTRA="--only-rat $ONLY_RAT"
fi

BSUB_CMD="bsub -W 4:00 -n 8 -gpu \"num=1\" -q gpu_a100 -P johnson \
    -J fit_green \
    -o $LOG \
    bash $REPO_DIR/scripts/cluster_fit_green.sh --run --weight $TARGET_WEIGHT --output-dir $OUTPUT_DIR $BSUB_EXTRA"

echo "Green Rat Body Model Fitting"
echo "  Green:   $GREEN_DIR"
echo "  Base:    $BASE_MODEL"
echo "  Output:  $OUTPUT_DIR"
echo "  Weight:  $TARGET_WEIGHT"
echo "  Log:     $LOG"
if [[ -n "$ONLY_RAT" ]]; then
    echo "  Rat:     $ONLY_RAT (single-animal test)"
fi
echo ""

if $DRY_RUN; then
    echo "[dry-run] $BSUB_CMD"
else
    eval "$BSUB_CMD"
    echo "Monitor: bjobs -w | grep fit_green"
    echo "Log:     tail -f $LOG"
fi
