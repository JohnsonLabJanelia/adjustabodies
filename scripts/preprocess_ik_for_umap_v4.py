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
import struct
import sys
import time
import numpy as np


def main():
    parser = argparse.ArgumentParser(description="Preprocess IK qpos for UMAP")
    parser.add_argument('--qpos-csv', required=True, help="Merged qpos_v4.csv")
    parser.add_argument('--green-dir', required=True,
                        help="Green data dir (for 3D trajectory → COM speed)")
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--traj', default='repaired_traj3d.bin')
    parser.add_argument('--max-residual', type=float, default=15.0,
                        help="Exclude frames with IK residual above this (mm)")
    parser.add_argument('--smooth-window', type=int, default=3,
                        help="Smoothing window for qpos before encoding (0=none)")
    parser.add_argument('--include-qvel', action='store_true',
                        help="Include joint velocities (dq/dt) in features")
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

    # ── Compute COM speed from 3D trajectory ────────────────────
    # Load the actual 3D keypoint data and compute 2D COM translational speed
    COM_KPS = [3, 4, 5, 6, 10]  # Neck, SpineL, TailBase, ShoulderL, ShoulderR
    FPS = 180
    print("  Computing COM speed from 3D trajectory...")

    traj_path = os.path.join(args.green_dir, args.traj)
    traj_data = np.memmap(traj_path, dtype=np.uint8, mode='r')
    n_traj_trials = struct.unpack_from('<I', traj_data, 8)[0]
    n_traj_fields = struct.unpack_from('<I', traj_data, 12)[0]
    # Parse field descriptors to find traj3d
    cumulative = 0
    traj3d_field = None
    for i in range(n_traj_fields):
        pos = 32 + i * 44
        name = bytes(traj_data[pos:pos+32]).split(b'\0')[0].decode()
        epf = struct.unpack_from('<I', traj_data, pos + 32)[0]
        esz = struct.unpack_from('<I', traj_data, pos + 36)[0]
        if name == 'traj3d':
            traj3d_field = {'epf': epf, 'esz': esz, 'offset': cumulative}
        cumulative += epf * esz
    # Parse index
    desc_end = 32 + n_traj_fields * 44
    idx_start = (desc_end + 7) & ~7
    traj_index = []
    for i in range(n_traj_trials):
        pos = idx_start + i * 12
        off = struct.unpack_from('<Q', traj_data, pos)[0]
        nf = struct.unpack_from('<I', traj_data, pos + 8)[0]
        traj_index.append((off, nf))

    # Precompute per-trial COM (XY only) for speed
    # Build a lookup: (trial_id, frame) → COM XY position
    speed = np.zeros(n_total, dtype=np.float32)
    prev_tid = -1
    prev_com = None
    for i in range(n_total):
        tid = trial_ids[i]
        frame = frame_ids[i]

        if tid != prev_tid:
            # Load this trial's 3D data and compute COM for all frames
            if tid < len(traj_index):
                t_off, t_nf = traj_index[tid]
                stride = traj3d_field['epf']
                start = t_off + t_nf * traj3d_field['offset']
                nbytes = t_nf * stride * 4
                arr = np.frombuffer(traj_data[start:start + nbytes],
                                     dtype=np.float32).reshape(t_nf, stride)
                # arr is (nf, 75) → reshape to (nf, 25, 3)
                kp3d = arr[:, :75].reshape(t_nf, 25, 3)
                # COM from selected keypoints (XY only, in mm)
                trial_com = np.nanmean(kp3d[:, COM_KPS, :2], axis=1)  # (nf, 2)
            else:
                trial_com = None
            prev_tid = tid
            prev_com = None

        if trial_com is not None and frame < len(trial_com):
            com_xy = trial_com[frame]
            if prev_com is not None and np.isfinite(com_xy).all() and np.isfinite(prev_com).all():
                speed[i] = np.sqrt(np.sum((com_xy - prev_com)**2))
            prev_com = com_xy
        else:
            prev_com = None

    print(f"  COM speed: mean={speed[speed>0].mean():.2f} mm/f, "
          f"max={speed.max():.1f} mm/f")

    # ── Smooth qpos per trial ────────────────────────────────────
    if args.smooth_window > 1:
        print(f"  Smoothing qpos (window={args.smooth_window}) per trial...")
        unique_tids = np.unique(trial_ids)
        for tid in unique_tids:
            mask = trial_ids == tid
            idx = np.where(mask)[0]
            if len(idx) < args.smooth_window:
                continue
            for j in range(n_hinges):
                col = hinges[idx, j]
                win = args.smooth_window
                if win % 2 == 0:
                    win += 1
                pad = win // 2
                padded = np.pad(col, (pad, pad), mode='edge')
                kernel = np.ones(win, dtype=np.float32) / win
                hinges[idx, j] = np.convolve(padded, kernel, mode='valid')

    # ── Compute qvel (joint velocities) per trial ─────────────────
    qvel = None
    if args.include_qvel:
        print(f"  Computing joint velocities (dq/dt)...")
        qvel = np.zeros_like(hinges)
        unique_tids = np.unique(trial_ids)
        for tid in unique_tids:
            mask = trial_ids == tid
            idx = np.where(mask)[0]
            if len(idx) < 2:
                continue
            qvel[idx] = np.gradient(hinges[idx], axis=0).astype(np.float32)

    # ── Sin/cos encoding + optional qvel ──────────────────────────
    sincos = np.concatenate([np.sin(hinges), np.cos(hinges)], axis=1).astype(np.float32)
    if qvel is not None:
        features = np.concatenate([sincos, qvel], axis=1).astype(np.float32)
        print(f"  Features: sin/cos({n_hinges}×2) + qvel({n_hinges}) = {features.shape[1]}D")
    else:
        features = sincos
        print(f"  Features: sin/cos({n_hinges}×2) = {features.shape[1]}D")

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
