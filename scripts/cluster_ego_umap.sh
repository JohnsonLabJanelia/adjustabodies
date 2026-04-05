#!/bin/bash
# cluster_ego_umap.sh — Egocentric 3D keypoint UMAP pipeline
#
# 3 stages:
#   1. Preprocess: egocenter keypoints, trim to analysis windows (CPU)
#   2. UMAP: PCA → GPU UMAP embedding (GPU, RAPIDS/cuML)
#   3. Pack: convert to Green binary format (CPU)
#
# Usage:
#   bash scripts/cluster_ego_umap.sh [--dry-run] [--keypoints all24|feet_only|...]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"

GREEN_DIR="/groups/johnson/johnsonlab/virtual_rodent/green"
UMAP_DIR="$GREEN_DIR/umap_ego"
OUTPUT_BIN="$GREEN_DIR/green_umap.bin"
LOG_PRE="$GREEN_DIR/ego_umap_preprocess.log"
LOG_UMAP="$GREEN_DIR/ego_umap_embed.log"
LOG_PACK="$GREEN_DIR/ego_umap_pack.log"

KEYPOINTS="all24"
N_PCA=20
N_NEIGHBORS=30
MIN_DIST=0.05
DRY_RUN=false
RUN_STAGE=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --dry-run)     DRY_RUN=true; shift ;;
        --keypoints)   KEYPOINTS="$2"; shift 2 ;;
        --n-pca)       N_PCA="$2"; shift 2 ;;
        --n-neighbors) N_NEIGHBORS="$2"; shift 2 ;;
        --min-dist)    MIN_DIST="$2"; shift 2 ;;
        --run-preprocess) RUN_STAGE="preprocess"; shift ;;
        --run-umap)    RUN_STAGE="umap"; shift ;;
        --run-pack)    RUN_STAGE="pack"; shift ;;
        *) echo "Unknown: $1"; exit 1 ;;
    esac
done

# ── Run mode (executed by bsub) ──────────────────────────────────────

if [[ "$RUN_STAGE" == "preprocess" ]]; then
    echo "=== Egocentric UMAP: Preprocessing ==="
    echo "Host: $(hostname), Date: $(date)"

    source ~/miniconda3/bin/activate && conda activate mjx
    pip install -e "$REPO_DIR" duckdb --quiet 2>&1 | tail -3

    python3 "$REPO_DIR/scripts/preprocess_ego_umap.py" \
        --green-dir "$GREEN_DIR" \
        --output-dir "$UMAP_DIR" \
        --keypoints "$KEYPOINTS"

    echo "=== Done ==="
    exit 0
fi

if [[ "$RUN_STAGE" == "umap" ]]; then
    echo "=== Egocentric UMAP: GPU Embedding ==="
    echo "Host: $(hostname), Date: $(date)"

    source ~/miniconda3/bin/activate && conda activate rapids

    python3 "$REPO_DIR/scripts/gpu_umap_ego.py" \
        --data-dir "$UMAP_DIR" \
        --n-pca "$N_PCA" \
        --n-neighbors "$N_NEIGHBORS" \
        --min-dist "$MIN_DIST"

    echo "=== Done ==="
    exit 0
fi

if [[ "$RUN_STAGE" == "pack" ]]; then
    echo "=== Egocentric UMAP: Packing Binary ==="
    echo "Host: $(hostname), Date: $(date)"

    source ~/miniconda3/bin/activate && conda activate mjx

    python3 "$REPO_DIR/scripts/pack_ego_umap_binary.py" \
        --umap-dir "$UMAP_DIR" \
        --output "$OUTPUT_BIN"

    echo "=== Done ==="
    exit 0
fi

# ── Submit mode ──────────────────────────────────────────────────────

echo "Egocentric UMAP Pipeline"
echo "  Green:      $GREEN_DIR"
echo "  UMAP dir:   $UMAP_DIR"
echo "  Output:     $OUTPUT_BIN"
echo "  Keypoints:  $KEYPOINTS"
echo "  PCA:        $N_PCA, UMAP: nn=$N_NEIGHBORS md=$MIN_DIST"
echo ""

# Stage 1: Preprocess (CPU, ~5 min)
BSUB_PRE="bsub -W 1:00 -n 4 -q local -P johnson \
    -J ego_pre \
    -o $LOG_PRE \
    bash $REPO_DIR/scripts/cluster_ego_umap.sh --run-preprocess --keypoints $KEYPOINTS"

# Stage 2: UMAP (GPU, ~10 min, depends on stage 1)
BSUB_UMAP="bsub -W 2:00 -n 4 -gpu \"num=1\" -q gpu_a100 -P johnson \
    -J ego_umap -w \"done(ego_pre)\" \
    -o $LOG_UMAP \
    bash $REPO_DIR/scripts/cluster_ego_umap.sh --run-umap --n-pca $N_PCA --n-neighbors $N_NEIGHBORS --min-dist $MIN_DIST"

# Stage 3: Pack (CPU, ~2 min, depends on stage 2)
BSUB_PACK="bsub -W 0:30 -n 1 -q local -P johnson \
    -J ego_pack -w \"done(ego_umap)\" \
    -o $LOG_PACK \
    bash $REPO_DIR/scripts/cluster_ego_umap.sh --run-pack"

if $DRY_RUN; then
    echo "[dry-run] Stage 1: $BSUB_PRE"
    echo ""
    echo "[dry-run] Stage 2: $BSUB_UMAP"
    echo ""
    echo "[dry-run] Stage 3: $BSUB_PACK"
else
    echo "Submitting 3-stage pipeline..."
    eval "$BSUB_PRE"
    eval "$BSUB_UMAP"
    eval "$BSUB_PACK"
    echo ""
    echo "Monitor: bjobs -w | grep ego"
    echo "Logs:"
    echo "  tail -f $LOG_PRE"
    echo "  tail -f $LOG_UMAP"
    echo "  tail -f $LOG_PACK"
fi
