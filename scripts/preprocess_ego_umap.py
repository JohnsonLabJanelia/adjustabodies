#!/usr/bin/env python3
"""Preprocess egocentric 3D keypoints for UMAP embedding.

Loads green_traj3d.bin, egocenters all frames (COM-center + heading-align),
trims to analysis windows, and saves .npy arrays for GPU UMAP.

This bypasses IK entirely — uses raw 3D keypoints in body-centered frame.

Usage:
    python3 scripts/preprocess_ego_umap.py \
        --green-dir /path/to/green \
        --output-dir /path/to/umap_ego

Outputs:
    ego_features.npy     — (N, D) egocentric keypoint features (float32)
    com_speed.npy        — (N,) COM translational speed in mm/frame (float32)
    trial_index.npz      — trial_ids, starts, lengths for binary packing
    metadata.npz         — fps, num_kp, feature_names
"""

import argparse
import os
import struct
import sys
import time
import numpy as np

# Add green scripts to path for full_trial_data
GREEN_SCRIPTS = os.path.expanduser("~/src/green/scripts/fitting")
if os.path.isdir(GREEN_SCRIPTS):
    sys.path.insert(0, GREEN_SCRIPTS)


# ── Constants ──────────────────────────────────────────────────────────

FPS = 180
NUM_RAT_KEYPOINTS = 24
NUM_KEYPOINTS = 25
STRIDE_3D = NUM_KEYPOINTS * 3

KP_SNOUT = 0
KP_NECK = 3
KP_SPINEL = 4
KP_TAILBASE = 5
KP_SHOULDERL = 6
KP_SHOULDERR = 10

COM_KEYPOINTS = [KP_NECK, KP_SPINEL, KP_TAILBASE, KP_SHOULDERL, KP_SHOULDERR]
HEADING_FRONT = KP_SNOUT
HEADING_BACK = KP_NECK

# Keypoint subsets for different UMAP feature configurations
KEYPOINT_SETS = {
    'all24': list(range(24)),          # Full skeleton: 24 × 3 = 72D
    'no_tail': list(range(20)),        # Skip tail (20-23): 20 × 3 = 60D
    'limbs_spine': [0, 3, 4, 5,       # Snout, Neck, SpineL, TailBase
                    6, 7, 8, 9,        # Left arm
                    10, 11, 12, 13,    # Right arm
                    14, 15, 16,        # Left leg
                    17, 18, 19],       # Right leg — 19 × 3 = 57D
    'feet_only': [9, 13, 16, 19],     # HandL/R, FootL/R: 4 × 3 = 12D
}


# ── Binary reader ──────────────────────────────────────────────────────

MAGIC = 0x024E5247
FORMAT_VERSION = 2
FIELD_NAME_LEN = 32
FIELD_DESC_SIZE = 44
INDEX_ENTRY_SIZE = 12


def read_green_binary(path, field_name='traj3d'):
    """Read green binary, return per-trial arrays."""
    data = np.memmap(path, dtype=np.uint8, mode='r')

    magic = struct.unpack_from('<I', data, 0)[0]
    assert magic == MAGIC, f"Bad magic: {magic:#x}"
    version = struct.unpack_from('<I', data, 4)[0]
    assert version == FORMAT_VERSION

    num_trials = struct.unpack_from('<I', data, 8)[0]
    num_fields = struct.unpack_from('<I', data, 12)[0]
    fps = struct.unpack_from('<I', data, 20)[0]

    fields = []
    cumulative = 0
    for i in range(num_fields):
        pos = 32 + i * FIELD_DESC_SIZE
        name = bytes(data[pos:pos + FIELD_NAME_LEN]).split(b'\x00')[0].decode()
        epf = struct.unpack_from('<I', data, pos + 32)[0]
        esz = struct.unpack_from('<I', data, pos + 36)[0]
        dtype_code = struct.unpack_from('<I', data, pos + 40)[0]
        fields.append({
            'name': name, 'epf': epf, 'esz': esz,
            'dtype': np.float32 if dtype_code == 0 else np.uint8,
            'byte_offset': cumulative,
        })
        cumulative += epf * esz

    fld = next(f for f in fields if f['name'] == field_name)

    desc_end = 32 + num_fields * FIELD_DESC_SIZE
    index_start = (desc_end + 7) & ~7
    index = []
    for i in range(num_trials):
        pos = index_start + i * INDEX_ENTRY_SIZE
        offset = struct.unpack_from('<Q', data, pos)[0]
        nf = struct.unpack_from('<I', data, pos + 8)[0]
        index.append((offset, nf))

    trials = []
    for trial_offset, nf in index:
        start = trial_offset + nf * fld['byte_offset']
        nbytes = nf * fld['epf'] * fld['esz']
        arr = np.frombuffer(data[start:start + nbytes], dtype=fld['dtype'])
        trials.append(arr.reshape(nf, fld['epf']))

    return trials, {'num_trials': num_trials, 'fps': fps}


# ── Egocentric transform ──────────────────────────────────────────────

def egocenter_trial(kp3d_flat):
    """Egocenter a single trial's keypoints.

    Args:
        kp3d_flat: (T, 75) raw keypoint data from binary

    Returns:
        ego: (T, 24, 3) egocentric keypoints
        com: (T, 3) center of mass
        speed: (T,) COM speed in mm/frame
        valid_mask: (T,) bool — True if enough keypoints are valid
    """
    T = kp3d_flat.shape[0]
    kp3d = kp3d_flat.reshape(T, NUM_KEYPOINTS, 3)

    # Strict quality check: ALL 24 rat keypoints must be valid (finite, non-zero)
    # This filters out frames with partial tracking or poor pose repair
    rat_kp = kp3d[:, :NUM_RAT_KEYPOINTS, :]  # (T, 24, 3)
    kp_finite = np.isfinite(rat_kp).all(axis=-1)     # (T, 24)
    kp_nonzero = ~(rat_kp == 0).all(axis=-1)          # (T, 24)
    valid_mask = (kp_finite & kp_nonzero).all(axis=1)  # (T,) — all 24 must pass

    # COM
    com_kps = kp3d[:, COM_KEYPOINTS, :]  # (T, 5, 3)
    com = np.nanmean(com_kps, axis=1)     # (T, 3)

    # COM speed (mm/frame)
    com_diff = np.diff(com, axis=0)
    speed = np.zeros(T, dtype=np.float32)
    speed[1:] = np.sqrt(np.sum(com_diff**2, axis=-1))

    # Heading angle
    front = kp3d[:, HEADING_FRONT, :2]
    back = kp3d[:, HEADING_BACK, :2]
    direction = front - back
    heading = np.arctan2(direction[:, 1], direction[:, 0])

    # Center on COM
    centered = kp3d[:, :NUM_RAT_KEYPOINTS, :].copy()
    centered -= com[:, None, :]

    # Rotate to heading-aligned frame
    cos_h = np.cos(-heading)
    sin_h = np.sin(-heading)

    ego = np.zeros_like(centered)
    ego[:, :, 0] = centered[:, :, 0] * cos_h[:, None] - centered[:, :, 1] * sin_h[:, None]
    ego[:, :, 1] = centered[:, :, 0] * sin_h[:, None] + centered[:, :, 1] * cos_h[:, None]
    ego[:, :, 2] = centered[:, :, 2]

    return ego, com, speed, valid_mask


# ── Main ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Preprocess egocentric keypoints for UMAP")
    parser.add_argument('--green-dir', required=True,
                        help="Green data directory (contains green_traj3d.bin, green.duckdb)")
    parser.add_argument('--output-dir', required=True,
                        help="Output directory for .npy files")
    parser.add_argument('--traj', default='repaired_traj3d.bin',
                        help="Trajectory binary filename")
    parser.add_argument('--keypoints', default='all24',
                        choices=list(KEYPOINT_SETS.keys()),
                        help="Which keypoints to include in features")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    kp_indices = KEYPOINT_SETS[args.keypoints]
    n_kp = len(kp_indices)
    n_features = n_kp * 3
    print(f"Keypoint set: {args.keypoints} ({n_kp} keypoints, {n_features}D)")

    # ── Load analysis windows from computed_metrics.csv ──
    # This uses the same trial_start_frame → trial_end_frame as the Green app playback.
    import pandas as pd
    metrics_path = os.path.join(args.green_dir, 'computed_metrics.csv')
    if not os.path.exists(metrics_path):
        print(f"ERROR: {metrics_path} not found. Run 'Process Dataset' in Green first.")
        sys.exit(1)

    metrics = pd.read_csv(metrics_path)
    print(f"  Loaded {len(metrics)} trials from computed_metrics.csv")

    # Also get animal IDs from DuckDB
    db_path = os.path.join(args.green_dir, 'green.duckdb')
    import duckdb
    con = duckdb.connect(db_path, read_only=True)
    animal_rows = con.execute("SELECT id, animal FROM trials ORDER BY id").fetchall()
    animal_map = {r[0]: r[1] for r in animal_rows}
    con.close()

    trial_windows = {}
    for _, row in metrics.iterrows():
        tid = int(row['trial_id'])
        start = row.get('trial_start_frame')
        end = row.get('trial_end_frame')
        if pd.notna(start) and pd.notna(end):
            start, end = int(start), int(end)
            if end > start + 10:
                trial_windows[tid] = (start, end)

    print(f"  {len(trial_windows)} trials with analysis windows")

    # ── Load trajectories ──
    traj_path = os.path.join(args.green_dir, args.traj)
    print(f"Loading: {traj_path}")
    trials_3d, meta = read_green_binary(traj_path, field_name='traj3d')
    print(f"  {meta['num_trials']} trials, {meta['fps']} fps")

    # ── Process all trials ──
    t0 = time.time()
    all_features = []
    all_speeds = []
    all_frame_ids = []
    all_animals = []
    trial_ids = []
    trial_starts = []
    trial_lengths = []
    frame_cursor = 0

    n_invalid = 0
    n_total_frames = 0

    for tid in sorted(trial_windows.keys()):
        if tid >= len(trials_3d):
            continue

        start_f, end_f = trial_windows[tid]
        kp3d_flat = trials_3d[tid]  # (T, 75)

        if end_f > kp3d_flat.shape[0]:
            end_f = kp3d_flat.shape[0]
        if end_f <= start_f + 10:
            continue

        # Egocenter the full trial first (for COM smoothness)
        ego, com, speed, valid_mask = egocenter_trial(kp3d_flat)

        # Trim to analysis window
        ego_win = ego[start_f:end_f]           # (W, 24, 3)
        speed_win = speed[start_f:end_f]       # (W,)
        valid_win = valid_mask[start_f:end_f]  # (W,)

        # Select keypoint subset and flatten
        features = ego_win[:, kp_indices, :].reshape(-1, n_features)  # (W, D)

        # Replace invalid frames with NaN (will be handled by UMAP)
        features[~valid_win] = np.nan
        n_invalid += (~valid_win).sum()
        n_total_frames += len(valid_win)

        # Also check for NaN in individual keypoints
        any_nan = np.isnan(features).any(axis=1)
        features[any_nan] = np.nan

        all_features.append(features.astype(np.float32))
        all_speeds.append(speed_win.astype(np.float32))

        # Global frame IDs for cursor mapping
        all_frame_ids.append(np.arange(start_f, end_f, dtype=np.int32))

        animal = animal_map.get(tid, 'unknown')
        all_animals.extend([animal] * len(features))

        trial_ids.append(tid)
        trial_starts.append(frame_cursor)
        trial_lengths.append(len(features))
        frame_cursor += len(features)

    elapsed = time.time() - t0
    total_frames = sum(trial_lengths)
    print(f"\nProcessed {len(trial_ids)} trials, {total_frames} frames in {elapsed:.1f}s")
    print(f"  Invalid frames: {n_invalid}/{n_total_frames} ({100*n_invalid/max(n_total_frames,1):.1f}%)")

    # ── Stack and save ──
    features_all = np.vstack(all_features)   # (N, D)
    speeds_all = np.concatenate(all_speeds)  # (N,)

    # Build animal ID array (uint8 encoded)
    animal_names = sorted(set(all_animals))
    animal_to_id = {name: i for i, name in enumerate(animal_names)}
    animal_ids = np.array([animal_to_id[a] for a in all_animals], dtype=np.uint8)

    # Valid mask: frames with no NaN
    valid_all = ~np.isnan(features_all).any(axis=1)
    print(f"  Valid frames for UMAP: {valid_all.sum()}/{len(valid_all)} "
          f"({100*valid_all.sum()/len(valid_all):.1f}%)")

    # Save
    frame_ids_all = np.concatenate(all_frame_ids)  # (N,) global frame numbers

    np.save(os.path.join(args.output_dir, 'ego_features.npy'), features_all)
    np.save(os.path.join(args.output_dir, 'com_speed.npy'), speeds_all)
    np.save(os.path.join(args.output_dir, 'animal_ids.npy'), animal_ids)
    np.save(os.path.join(args.output_dir, 'valid_mask.npy'), valid_all)
    np.save(os.path.join(args.output_dir, 'frame_ids.npy'), frame_ids_all)

    np.savez(os.path.join(args.output_dir, 'trial_index.npz'),
             trial_ids=np.array(trial_ids, dtype=np.int32),
             starts=np.array(trial_starts, dtype=np.int64),
             lengths=np.array(trial_lengths, dtype=np.int32))

    np.savez(os.path.join(args.output_dir, 'metadata.npz'),
             fps=FPS, num_kp=n_kp, n_features=n_features,
             keypoint_set=args.keypoints,
             keypoint_indices=np.array(kp_indices),
             animal_names=np.array(animal_names),
             total_frames=total_frames,
             valid_frames=int(valid_all.sum()))

    print(f"\nSaved to {args.output_dir}/:")
    print(f"  ego_features.npy  — {features_all.shape} ({features_all.nbytes/1e6:.1f} MB)")
    print(f"  com_speed.npy     — {speeds_all.shape}")
    print(f"  animal_ids.npy    — {animal_ids.shape}")
    print(f"  valid_mask.npy    — {valid_all.shape}")
    print(f"  trial_index.npz")
    print(f"  metadata.npz")


if __name__ == "__main__":
    main()
