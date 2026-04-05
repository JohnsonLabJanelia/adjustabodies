#!/bin/bash
# cluster_umap_pipeline.sh — Full UMAP pipeline: preprocess + embed
#
# Two-stage pipeline:
#   Stage 1 (mjx env): Parse qpos CSV, compute qvel, save .npy arrays
#   Stage 2 (rapids env): GPU UMAP on three feature sets
#
# Usage:
#   bash scripts/cluster_umap_pipeline.sh [--dry-run] [--stage 1|2|both]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"

GREEN_DIR="/groups/johnson/johnsonlab/virtual_rodent/green"
QPOS_CSV="$GREEN_DIR/qpos_export_mjx.csv"
MODEL="/groups/johnson/johnsonlab/virtual_rodent/body_model/rodent_red_mj_dev_fitted.mjb"
GREEN_DB="$GREEN_DIR/green.duckdb"
UMAP_DIR="$GREEN_DIR/umap_data"
LOG_PREFIX="$GREEN_DIR/umap"

DRY_RUN=false
RUN_STAGE=""
STAGE="both"

while [[ $# -gt 0 ]]; do
    case $1 in
        --dry-run)    DRY_RUN=true; shift ;;
        --run-stage)  RUN_STAGE="$2"; shift 2 ;;
        --stage)      STAGE="$2"; shift 2 ;;
        *) echo "Unknown: $1"; exit 1 ;;
    esac
done

# ── Run mode (inside job) ──────────────────────────────────────
if [[ "$RUN_STAGE" == "1" ]]; then
    echo "=== UMAP Stage 1: Preprocessing ==="
    echo "Host: $(hostname), Date: $(date)"
    source ~/miniconda3/bin/activate && conda activate mjx
    pip install -e "$REPO_DIR" duckdb scipy --quiet 2>&1 | tail -3

    python3 "$REPO_DIR/scripts/preprocess_ik_for_umap.py" \
        --qpos-csv "$QPOS_CSV" \
        --model "$MODEL" \
        --green-db "$GREEN_DB" \
        --output-dir "$UMAP_DIR" \
        --fps 180 \
        --smooth-window 37

    echo "=== Stage 1 Done ==="
    exit 0
fi

if [[ "$RUN_STAGE" == "2" ]]; then
    echo "=== UMAP Stage 2: GPU Embedding ==="
    echo "Host: $(hostname), Date: $(date)"
    source ~/miniconda3/bin/activate && conda activate rapids

    python3 "$REPO_DIR/scripts/gpu_umap.py" \
        --data-dir "$UMAP_DIR" \
        --n-pca-pos 20 \
        --n-pca-vel 20 \
        --n-pca-combined 30 \
        --n-neighbors 15 \
        --min-dist 0.1

    echo "=== Stage 2 Done ==="
    exit 0
fi

# ── Submit mode ────────────────────────────────────────────────

echo "UMAP Pipeline for Green Dataset"
echo "  Input:  $QPOS_CSV"
echo "  Model:  $MODEL"
echo "  Output: $UMAP_DIR"
echo ""

if [[ "$STAGE" == "1" || "$STAGE" == "both" ]]; then
    BSUB_1="bsub -W 2:00 -n 8 -q local -P johnson \
        -J umap_preprocess \
        -o ${LOG_PREFIX}_preprocess.log \
        bash $REPO_DIR/scripts/cluster_umap_pipeline.sh --run-stage 1"

    echo "Stage 1: Preprocess (CPU, mjx env)"
    if $DRY_RUN; then
        echo "  [dry-run] $BSUB_1"
    else
        JOB1=$(eval "$BSUB_1" 2>&1 | grep -oP '(?<=<)\d+(?=>)')
        echo "  Job $JOB1 submitted"
    fi
fi

if [[ "$STAGE" == "2" || "$STAGE" == "both" ]]; then
    DEP_FLAG=""
    if [[ "$STAGE" == "both" && -n "${JOB1:-}" ]]; then
        DEP_FLAG="-w \"done($JOB1)\""
    fi

    BSUB_2="bsub -W 1:00 -n 8 -gpu \"num=1\" -q gpu_a100 -P johnson \
        $DEP_FLAG \
        -J umap_embed \
        -o ${LOG_PREFIX}_embed.log \
        bash $REPO_DIR/scripts/cluster_umap_pipeline.sh --run-stage 2"

    echo "Stage 2: GPU UMAP (A100, rapids env)"
    if $DRY_RUN; then
        echo "  [dry-run] $BSUB_2"
    else
        eval "$BSUB_2"
    fi
fi

echo ""
echo "Monitor: bjobs -w | grep umap"
echo "Logs:    ${LOG_PREFIX}_preprocess.log / ${LOG_PREFIX}_embed.log"
echo "Output:  $UMAP_DIR/"
