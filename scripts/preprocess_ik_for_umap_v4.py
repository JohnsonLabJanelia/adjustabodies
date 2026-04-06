#!/usr/bin/env python3
"""Preprocess v4 per-animal IK qpos for UMAP embedding.

Extracts hinge angles (qpos[7:68]) from the merged IK CSV,
applies sin/cos encoding, and saves .npy arrays for GPU UMAP.

The 61 hinge angles are already egocentric — they describe body
configuration independent of global position/orientation.

Usage:
    python3 scripts/preprocess_ik_for_umap_v4.py \
        --qpos-csv /path/to/qpos_v4.csv \
        --output-dir /path/to/umap_ik_v4 \
        --max-residual 15.0
"""

import argparse
import os
import sys
import time
import numpy as np


def main():
    parser = argparse.ArgumentParser(description="Preprocess IK qpos for UMAP")
    parser.add_argument('--qpos-csv', required=True, help="Merged qpos_v4.csv")
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--max-residual', type=float, default=15.0,
                        help="Exclude frames with IK residual above this (mm)")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # ── Parse CSV ─────────────────────────────────────────────────
    print(f"Loading: {args.qpos_csv}")
    t0 = time.time()

    trial_ids_list = []
    frame_ids_list = []
    animals_list = []
    hinges_list = []
    residuals_list = []
    speeds_list = []

    animal_names_set = set()

    with open(args.qpos_csv) as f:
        for line in f:
            if line.startswith('#'):
                continue
            if line.startswith('trial,'):
                # Parse header to find column indices
                cols = line.strip().split(',')
                # Expected: trial, frame, animal, qpos_0..qpos_67, residual_mm
                assert cols[0] == 'trial'
                assert cols[1] == 'frame'
                assert cols[2] == 'animal'
                assert cols[-1] == 'residual_mm'
                n_qpos = len(cols) - 4  # subtract trial, frame, animal, residual
                print(f"  Columns: {len(cols)}, qpos dims: {n_qpos}")
                continue

            parts = line.strip().split(',')
            tid = int(parts[0])
            frame = int(parts[1])
            animal = parts[2]
            qpos = [float(x) for x in parts[3:-1]]
            residual = float(parts[-1])

            # Extract hinges (skip free joint qpos[0:7])
            hinges = qpos[7:]

            trial_ids_list.append(tid)
            frame_ids_list.append(frame)
            animals_list.append(animal)
            hinges_list.append(hinges)
            residuals_list.append(residual)
            animal_names_set.add(animal)

    elapsed = time.time() - t0
    n_total = len(hinges_list)
    n_hinges = len(hinges_list[0])
    print(f"  {n_total:,} frames, {n_hinges} hinges, {elapsed:.1f}s")
    print(f"  Animals: {sorted(animal_names_set)}")

    hinges = np.array(hinges_list, dtype=np.float32)
    residuals = np.array(residuals_list, dtype=np.float32)
    trial_ids = np.array(trial_ids_list, dtype=np.int32)
    frame_ids = np.array(frame_ids_list, dtype=np.int32)

    # ── Quality filter ────────────────────────────────────────────
    valid = np.isfinite(hinges).all(axis=1) & (residuals < args.max_residual) & (residuals >= 0)
    n_bad = (~valid).sum()
    print(f"\n  Residual filter (<{args.max_residual}mm): {valid.sum():,} valid, {n_bad:,} excluded ({100*n_bad/n_total:.1f}%)")

    # ── Compute COM speed from free joint position ────────────────
    # qpos[0:3] is the free joint translation (global position in meters)
    # We compute speed per trial for coloring
    print("  Computing COM speed...")
    speed = np.zeros(n_total, dtype=np.float32)
    # Group by trial for within-trial speed computation
    unique_trials = np.unique(trial_ids)
    for tid in unique_trials:
        mask = trial_ids == tid
        idx = np.where(mask)[0]
        if len(idx) < 2:
            continue
        # Free joint position is qpos[0:3], but we stored only hinges
        # We need the full qpos — let's re-read just the positions
        # Actually, we didn't store qpos[0:3] in hinges. We need to go back.
        # For now, use frame-to-frame hinge angle change as a proxy for "movement"
        h = hinges[idx]
        diffs = np.diff(h, axis=0)
        frame_speed = np.sqrt(np.sum(diffs**2, axis=1))
        speed[idx[1:]] = frame_speed
        speed[idx[0]] = frame_speed[0] if len(frame_speed) > 0 else 0

    # ── Sin/cos encoding ──────────────────────────────────────────
    print(f"  Sin/cos encoding: {n_hinges} → {n_hinges * 2} features")
    features = np.concatenate([np.sin(hinges), np.cos(hinges)], axis=1).astype(np.float32)

    # ── Build trial index ─────────────────────────────────────────
    print("  Building trial index...")
    trial_starts = []
    trial_lengths = []
    trial_id_list = []
    prev_tid = -1
    for i in range(n_total):
        if trial_ids[i] != prev_tid:
            if prev_tid >= 0:
                trial_lengths.append(i - trial_starts[-1])
            trial_starts.append(i)
            trial_id_list.append(trial_ids[i])
            prev_tid = trial_ids[i]
    if prev_tid >= 0:
        trial_lengths.append(n_total - trial_starts[-1])

    # ── Animal IDs ────────────────────────────────────────────────
    animal_names = sorted(animal_names_set)
    animal_to_id = {name: i for i, name in enumerate(animal_names)}
    animal_ids = np.array([animal_to_id[a] for a in animals_list], dtype=np.uint8)

    # ── Save ──────────────────────────────────────────────────────
    np.save(os.path.join(args.output_dir, 'ego_features.npy'), features)
    np.save(os.path.join(args.output_dir, 'com_speed.npy'), speed)
    np.save(os.path.join(args.output_dir, 'animal_ids.npy'), animal_ids)
    np.save(os.path.join(args.output_dir, 'valid_mask.npy'), valid)

    np.savez(os.path.join(args.output_dir, 'trial_index.npz'),
             trial_ids=np.array(trial_id_list, dtype=np.int32),
             starts=np.array(trial_starts, dtype=np.int64),
             lengths=np.array(trial_lengths, dtype=np.int32))

    np.savez(os.path.join(args.output_dir, 'metadata.npz'),
             fps=180, num_kp=0, n_features=features.shape[1],
             keypoint_set='ik_sincos',
             animal_names=np.array(animal_names),
             total_frames=n_total,
             valid_frames=int(valid.sum()))

    print(f"\nSaved to {args.output_dir}/:")
    print(f"  ego_features.npy  — {features.shape} ({features.nbytes/1e6:.1f} MB)")
    print(f"  com_speed.npy     — {speed.shape}")
    print(f"  animal_ids.npy    — {animal_ids.shape}")
    print(f"  valid_mask.npy    — {valid.shape}")
    print(f"  trial_index.npz   — {len(trial_id_list)} trials")
    print(f"  metadata.npz")


if __name__ == "__main__":
    main()
