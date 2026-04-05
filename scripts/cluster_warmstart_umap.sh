#!/bin/bash
# cluster_warmstart_umap.sh — Full pipeline: preprocess warm-start qpos → UMAP → binary
#
# Runs after batch_ik_warmstart.py completes. Three stages:
#   Stage 1 (CPU): preprocess qpos_warmstart.csv → .npy arrays
#   Stage 2 (GPU): 3D UMAP × 3 embeddings + pack green_umap.bin
#
# Usage:
#   bash scripts/cluster_warmstart_umap.sh [--dry-run]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"

GREEN_DIR="/groups/johnson/johnsonlab/virtual_rodent/green"
QPOS_CSV="$GREEN_DIR/qpos_warmstart.csv"
MODEL="/groups/johnson/johnsonlab/virtual_rodent/body_model/rodent_red_mj_dev_fitted.mjb"
GREEN_DB="$GREEN_DIR/green.duckdb"
UMAP_DIR="$GREEN_DIR/umap_warmstart"
LOG_PREFIX="$GREEN_DIR/umap_warmstart"

DRY_RUN=false
RUN_STAGE=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --dry-run)    DRY_RUN=true; shift ;;
        --run-stage)  RUN_STAGE="$2"; shift 2 ;;
        *) echo "Unknown: $1"; exit 1 ;;
    esac
done

if [[ "$RUN_STAGE" == "1" ]]; then
    echo "=== Warm-Start UMAP Stage 1: Preprocessing ==="
    echo "Host: $(hostname), Date: $(date)"
    source ~/miniconda3/bin/activate && conda activate mjx
    pip install -e "$REPO_DIR" duckdb scipy --quiet 2>&1 | tail -3

    python3 "$REPO_DIR/scripts/preprocess_ik_for_umap.py" \
        --qpos-csv "$QPOS_CSV" \
        --model "$MODEL" \
        --green-db "$GREEN_DB" \
        --metrics-csv "$GREEN_DIR/computed_metrics.csv" \
        --output-dir "$UMAP_DIR" \
        --fps 180 \
        --smooth-window 37

    echo "=== Stage 1 Done ==="
    exit 0
fi

if [[ "$RUN_STAGE" == "2" ]]; then
    echo "=== Warm-Start UMAP Stage 2: GPU Embedding + Pack ==="
    echo "Host: $(hostname), Date: $(date)"
    source ~/miniconda3/bin/activate && conda activate rapids

    python3 "$REPO_DIR/scripts/gpu_umap.py" \
        --data-dir "$UMAP_DIR" \
        --n-pca-pos 30 \
        --n-pca-vel 20 \
        --n-pca-combined 40 \
        --n-neighbors 50 \
        --min-dist 0.02

    echo ""
    echo "--- Packing green_umap.bin ---"
    python3 "$REPO_DIR/scripts/pack_umap_binary.py" \
        --umap-dir "$UMAP_DIR" \
        --output "$GREEN_DIR/green_umap.bin"

    echo "=== Stage 2 Done ==="
    exit 0
fi

# ── Submit mode ────────────────────────────────────────────────

echo "Warm-Start UMAP Pipeline"
echo "  Input:  $QPOS_CSV"
echo "  Output: $UMAP_DIR + $GREEN_DIR/green_umap.bin"
echo ""

# Check if warm-start IK is done
if [[ ! -f "$QPOS_CSV" ]]; then
    echo "WARNING: $QPOS_CSV not found yet. Submitting anyway (Stage 1 will wait)."
fi

BSUB_1="bsub -W 2:00 -n 8 -q local -P johnson \
    -J ws_umap_pre \
    -o ${LOG_PREFIX}_preprocess.log \
    bash $REPO_DIR/scripts/cluster_warmstart_umap.sh --run-stage 1"

echo "Stage 1: Preprocess (CPU)"
if $DRY_RUN; then
    echo "  [dry-run] $BSUB_1"
else
    JOB1=$(eval "$BSUB_1" 2>&1 | grep -oP '(?<=<)\d+(?=>)')
    echo "  Job $JOB1 submitted"
fi

BSUB_2="bsub -W 1:00 -n 8 -gpu 'num=1' -q gpu_a100 -P johnson \
    -w 'done($JOB1)' \
    -J ws_umap_gpu \
    -o ${LOG_PREFIX}_embed.log \
    bash $REPO_DIR/scripts/cluster_warmstart_umap.sh --run-stage 2"

echo "Stage 2: GPU UMAP + Pack (depends on Stage 1)"
if $DRY_RUN; then
    echo "  [dry-run] $BSUB_2"
else
    eval "$BSUB_2"
fi

echo ""
echo "Monitor: bjobs -w | grep ws_umap"
