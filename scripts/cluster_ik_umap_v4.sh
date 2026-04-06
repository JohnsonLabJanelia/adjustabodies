#!/bin/bash
# cluster_ik_umap_v4.sh — IK-based UMAP pipeline (sin/cos hinge angles)
#
# 3 stages:
#   1. Preprocess: extract hinges from qpos CSV, sin/cos encode (CPU)
#   2. UMAP: PCA → GPU UMAP embedding (GPU, RAPIDS/cuML)
#   3. Pack: convert to Green binary format (CPU)
#
# Usage:
#   bash scripts/cluster_ik_umap_v4.sh [--dry-run]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"

GREEN_DIR="/groups/johnson/johnsonlab/virtual_rodent/green"
QPOS_CSV="$GREEN_DIR/qpos_v4.csv"
UMAP_DIR="$GREEN_DIR/umap_ik_v4"
OUTPUT_BIN="$GREEN_DIR/green_umap_ik.bin"

N_PCA=40
N_NEIGHBORS=200
MIN_DIST=0.10
METRIC=cosine
DRY_RUN=false
RUN_STAGE=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --dry-run)       DRY_RUN=true; shift ;;
        --run-preprocess) RUN_STAGE="preprocess"; shift ;;
        --run-umap)      RUN_STAGE="umap"; shift ;;
        --run-pack)      RUN_STAGE="pack"; shift ;;
        *) echo "Unknown: $1"; exit 1 ;;
    esac
done

if [[ "$RUN_STAGE" == "preprocess" ]]; then
    echo "=== IK UMAP v4: Preprocessing ==="
    echo "Host: $(hostname), Date: $(date)"
    source ~/miniconda3/bin/activate && conda activate mjx
    pip install -e "$REPO_DIR" duckdb --quiet 2>&1 | tail -3 || true

    python3 "$REPO_DIR/scripts/preprocess_ik_for_umap_v4.py" \
        --qpos-csv "$QPOS_CSV" \
        --green-dir "$GREEN_DIR" \
        --output-dir "$UMAP_DIR" \
        --max-residual 15.0 \
        --smooth-window 3 \
        --include-qvel

    echo "=== Done ==="
    exit 0
fi

if [[ "$RUN_STAGE" == "umap" ]]; then
    echo "=== IK UMAP v4: GPU Embedding ==="
    echo "Host: $(hostname), Date: $(date)"
    source ~/miniconda3/bin/activate && conda activate rapids

    python3 "$REPO_DIR/scripts/gpu_umap_ego.py" \
        --data-dir "$UMAP_DIR" \
        --n-pca "$N_PCA" \
        --n-neighbors "$N_NEIGHBORS" \
        --min-dist "$MIN_DIST" \
        --metric "$METRIC"

    echo "=== Done ==="
    exit 0
fi

if [[ "$RUN_STAGE" == "pack" ]]; then
    echo "=== IK UMAP v4: Packing Binary ==="
    echo "Host: $(hostname), Date: $(date)"
    source ~/miniconda3/bin/activate && conda activate mjx

    python3 "$REPO_DIR/scripts/pack_ego_umap_binary.py" \
        --umap-dir "$UMAP_DIR" \
        --output "$OUTPUT_BIN"

    echo "=== Done ==="
    exit 0
fi

# ── Submit mode ──────────────────────────────────────────────────────

echo "IK UMAP v4 Pipeline (sin/cos hinge angles)"
echo "  Input:    $QPOS_CSV"
echo "  UMAP dir: $UMAP_DIR"
echo "  Output:   $OUTPUT_BIN"
echo "  PCA: $N_PCA, UMAP: nn=$N_NEIGHBORS md=$MIN_DIST metric=$METRIC"
echo ""

# Stage 1: Preprocess (CPU, ~5 min for 2.5M frames)
BSUB_PRE="bsub -W 1:00 -n 4 -q local -P johnson \
    -J ik_pre \
    -o $GREEN_DIR/ik_umap_preprocess.log \
    bash $REPO_DIR/scripts/cluster_ik_umap_v4.sh --run-preprocess"

# Stage 2: UMAP (GPU, ~5 min)
BSUB_UMAP="bsub -W 4:00 -n 4 -gpu \"num=1\" -q gpu_a100 -P johnson \
    -J ik_umap -w \"done(ik_pre)\" \
    -o $GREEN_DIR/ik_umap_embed.log \
    bash $REPO_DIR/scripts/cluster_ik_umap_v4.sh --run-umap"

# Stage 3: Pack (CPU, ~2 min)
BSUB_PACK="bsub -W 0:30 -n 1 -q local -P johnson \
    -J ik_pack -w \"done(ik_umap)\" \
    -o $GREEN_DIR/ik_umap_pack.log \
    bash $REPO_DIR/scripts/cluster_ik_umap_v4.sh --run-pack"

if $DRY_RUN; then
    echo "[dry-run] Stage 1: $BSUB_PRE"
    echo "[dry-run] Stage 2: $BSUB_UMAP"
    echo "[dry-run] Stage 3: $BSUB_PACK"
else
    echo "Submitting 3-stage pipeline..."
    eval "$BSUB_PRE"
    eval "$BSUB_UMAP"
    eval "$BSUB_PACK"
    echo ""
    echo "Monitor: bjobs -w | grep ik_"
fi
