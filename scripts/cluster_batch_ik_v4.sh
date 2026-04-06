#!/bin/bash
# cluster_batch_ik_v4.sh — Batch IK with per-animal v4 fitted models
#
# Submits 5 parallel jobs (one per animal), each using its own fitted model.
# Results merged into a single CSV at the end.
#
# Usage:
#   bash scripts/cluster_batch_ik_v4.sh [--dry-run]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"

GREEN_DIR="/groups/johnson/johnsonlab/virtual_rodent/green"
MODELS_DIR="/groups/johnson/johnsonlab/virtual_rodent/body_model/green_fits_v4"
OUTPUT_DIR="$GREEN_DIR/qpos_v4"
LOG_DIR="$GREEN_DIR"

DRY_RUN=false
RUN_ANIMAL=""
MERGE_MODE=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --dry-run)     DRY_RUN=true; shift ;;
        --run-animal)  RUN_ANIMAL="$2"; shift 2 ;;
        --merge)       MERGE_MODE=true; shift ;;
        *) echo "Unknown: $1"; exit 1 ;;
    esac
done

# ── Run mode: process one animal ─────────────────────────────────────

if [[ -n "$RUN_ANIMAL" ]]; then
    echo "=== Batch IK v4: $RUN_ANIMAL ==="
    echo "Host: $(hostname), Date: $(date)"

    source ~/miniconda3/bin/activate && conda activate mjx
    pip install -e "$REPO_DIR" duckdb --quiet 2>&1 | tail -3 || true

    MODEL="$MODELS_DIR/rodent_green_${RUN_ANIMAL}.mjb"
    OUTPUT="$OUTPUT_DIR/qpos_${RUN_ANIMAL}.csv"
    mkdir -p "$OUTPUT_DIR"

    python3 "$REPO_DIR/scripts/batch_ik_warmstart.py" \
        --green-dir "$GREEN_DIR" \
        --model "$MODEL" \
        --output "$OUTPUT" \
        --traj repaired_traj3d.bin \
        --animal "$RUN_ANIMAL" \
        --workers 8

    echo "=== Done: $RUN_ANIMAL ==="
    exit 0
fi

# ── Merge mode: combine per-animal CSVs ──────────────────────────────

if $MERGE_MODE; then
    echo "Merging per-animal CSVs (adding animal column)..."
    MERGED="$GREEN_DIR/qpos_v4.csv"

    # Write header
    echo "# GREEN v4 per-animal IK export (merged)" > "$MERGED"
    echo "# models: $MODELS_DIR" >> "$MERGED"
    # Get column header from first file, insert animal after frame
    FIRST_CSV=$(ls "$OUTPUT_DIR"/qpos_*.csv 2>/dev/null | head -1)
    ORIG_HEADER=$(grep -v '^#' "$FIRST_CSV" | head -1)
    # Insert 'animal' after 'frame' column: trial,frame,animal,qpos_0,...
    echo "$ORIG_HEADER" | sed 's/^trial,frame,/trial,frame,animal,/' >> "$MERGED"

    for animal in captain emilie heisenberg mario remy; do
        CSV="$OUTPUT_DIR/qpos_${animal}.csv"
        if [[ ! -f "$CSV" ]]; then
            echo "  WARNING: $CSV not found, skipping"
            continue
        fi
        # Skip comment and header lines, prepend animal after trial,frame
        grep -v '^#' "$CSV" | tail -n +2 | sed "s/^\([^,]*,[^,]*\),/\1,${animal},/" >> "$MERGED"
        echo "  $animal: $(grep -vc '^#' "$CSV") data lines"
    done
    echo "Merged: $MERGED ($(wc -l < "$MERGED") lines)"
    exit 0
fi

# ── Submit mode: launch 5 parallel jobs ──────────────────────────────

echo "Batch IK v4 — 5 parallel per-animal jobs"
echo "  Green:   $GREEN_DIR"
echo "  Models:  $MODELS_DIR"
echo "  Output:  $OUTPUT_DIR/"
echo ""

mkdir -p "$OUTPUT_DIR" 2>/dev/null || true

for animal in captain emilie heisenberg mario remy; do
    LOG="$LOG_DIR/batch_ik_v4_${animal}.log"
    BSUB_CMD="bsub -W 6:00 -n 8 -q local -P johnson \
        -R \"rusage[mem=16000]\" \
        -J ik_${animal} \
        -o $LOG \
        bash $REPO_DIR/scripts/cluster_batch_ik_v4.sh --run-animal $animal"

    if $DRY_RUN; then
        echo "[dry-run] $animal: $BSUB_CMD"
    else
        eval "$BSUB_CMD"
        echo "  $animal → $LOG"
    fi
done

if ! $DRY_RUN; then
    echo ""
    echo "Monitor: bjobs -w | grep ik_"
    echo "Merge when done: bash $REPO_DIR/scripts/cluster_batch_ik_v4.sh --merge"
fi
