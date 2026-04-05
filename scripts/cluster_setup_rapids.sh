#!/bin/bash
# cluster_setup_rapids.sh — Create rapids conda env on a compute node
#
# MUST be run from a compute node (bsub), not the login node.
#
# Usage:
#   # From login node, submit as interactive or batch:
#   bsub -W 1:00 -n 4 -gpu "num=1" -q gpu_a100 -P johnson -Is \
#       bash /groups/johnson/home/johnsonr/src/adjustabodies/scripts/cluster_setup_rapids.sh
#
#   # Or as batch job:
#   bash scripts/cluster_setup_rapids.sh [--dry-run]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"
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
    echo "=== Setting up rapids conda environment ==="
    echo "Host: $(hostname)"
    echo "Date: $(date)"

    source ~/miniconda3/bin/activate

    # Check if rapids env already exists
    if conda env list | grep -q "rapids"; then
        echo "rapids env already exists. Updating..."
        conda activate rapids
        conda install -c rapidsai -c conda-forge -c nvidia \
            cuml=26.02 python=3.12 cuda-version=12.6 -y
    else
        echo "Creating rapids env..."
        conda create -n rapids -c rapidsai -c conda-forge -c nvidia \
            cuml=26.02 python=3.12 cuda-version=12.6 -y
        conda activate rapids
    fi

    # Install additional packages
    pip install matplotlib h5py --quiet

    # Verify
    echo ""
    echo "--- Verification ---"
    python3 -c "
import cuml
print(f'cuML {cuml.__version__}')

import cupy as cp
print(f'CuPy {cp.__version__}')
print(f'GPU: {cp.cuda.runtime.getDeviceProperties(0)[\"name\"].decode()}')
print(f'VRAM: {cp.cuda.runtime.memGetInfo()[1] / 1e9:.1f} GB')

from cuml.manifold.umap import UMAP
from cuml.decomposition import PCA
import numpy as np

# Smoke test
X = np.random.randn(1000, 61).astype(np.float32)
pca = PCA(n_components=20)
X_pca = pca.fit_transform(cp.asarray(X))
umap = UMAP(n_components=2)
emb = umap.fit_transform(X_pca)
print(f'Smoke test: {X.shape} → PCA {X_pca.shape} → UMAP {emb.shape}  ✓')
"
    echo ""
    echo "=== rapids env ready ==="
    exit 0
fi

# Submit mode
LOG="/groups/johnson/johnsonlab/virtual_rodent/green/setup_rapids.log"

BSUB_CMD="bsub -W 1:00 -n 4 -gpu \"num=1\" -q gpu_a100 -P johnson \
    -J setup_rapids \
    -o $LOG \
    bash $REPO_DIR/scripts/cluster_setup_rapids.sh --run"

echo "Setup rapids conda env (compute node)"
echo "  Log: $LOG"
echo ""

if $DRY_RUN; then
    echo "[dry-run] $BSUB_CMD"
else
    eval "$BSUB_CMD"
    echo "Monitor: bjobs -w | grep setup"
    echo "Log:     $LOG"
fi
