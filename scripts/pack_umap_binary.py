#!/usr/bin/env python3
"""Pack UMAP .npy outputs into Green's TrajReader binary format.

Reads the UMAP embeddings and metadata from the preprocessing step,
packs them into green_umap.bin with the same trial/frame indexing
as the original trajectory data.

Output format: Green binary v2 (same as green_traj3d.bin)
  Fields per frame:
    umap_qpos      3 × float32   UMAP embedding of joint angles
    umap_qvel      3 × float32   UMAP embedding of joint velocities
    umap_combined   3 × float32   UMAP embedding of combined
    com_speed      1 × float32   COM translational speed (m/s)

Usage:
    python3 scripts/pack_umap_binary.py \
        --umap-dir /path/to/umap_data \
        --output /path/to/green_umap.bin
"""

import argparse
import os
import struct
import numpy as np


MAGIC = 0x024E5247           # "GRN\x02"
FORMAT_VERSION = 2
FIELD_NAME_LEN = 32
FIELD_DESC_SIZE = 44         # 32 + 4 + 4 + 4
INDEX_ENTRY_SIZE = 12        # 8 + 4
DTYPE_FLOAT32 = 0


def pack_field_desc(name, elements_per_frame, element_size, dtype_code):
    """Pack a field descriptor (44 bytes)."""
    name_bytes = name.encode('utf-8')[:FIELD_NAME_LEN]
    name_bytes = name_bytes + b'\x00' * (FIELD_NAME_LEN - len(name_bytes))
    return name_bytes + struct.pack('<III', elements_per_frame, element_size, dtype_code)


def main():
    parser = argparse.ArgumentParser(description="Pack UMAP outputs into Green binary")
    parser.add_argument('--umap-dir', required=True, help="Directory with .npy files")
    parser.add_argument('--output', required=True, help="Output .bin path (green_umap.bin)")
    args = parser.parse_args()

    # ── Load data ───────────────────────────────────────────────
    print(f"Loading from {args.umap_dir}/")

    # Trial index (maps flat frame array back to per-trial structure)
    tidx = np.load(os.path.join(args.umap_dir, 'trial_index.npz'))
    trial_ids = tidx['trial_ids']       # [n_trials]
    trial_starts = tidx['starts']       # [n_trials] int64
    trial_lengths = tidx['lengths']     # [n_trials] int32
    n_trials = len(trial_ids)
    total_frames = int(trial_lengths.sum())
    print(f"  {n_trials} trials, {total_frames} total frames")

    # Load valid mask (UMAP may have filtered NaN frames)
    valid_mask_path = os.path.join(args.umap_dir, 'umap_valid_mask.npy')
    valid_mask = None
    if os.path.exists(valid_mask_path):
        valid_mask = np.load(valid_mask_path)
        n_valid = int(valid_mask.sum())
        print(f"  Valid mask: {n_valid}/{total_frames} frames ({100*n_valid/total_frames:.1f}%)")

    # Load embeddings — if shorter than total_frames, expand using valid_mask
    def load_or_zeros(name, dims):
        path = os.path.join(args.umap_dir, f'umap_{name}.npy')
        if os.path.exists(path):
            arr = np.load(path).astype(np.float32)
            print(f"  umap_{name}: {arr.shape}")
            # Pad/truncate to exactly 3D
            if arr.shape[1] < dims:
                arr = np.concatenate([arr, np.zeros((arr.shape[0], dims - arr.shape[1]), dtype=np.float32)], axis=1)
            elif arr.shape[1] > dims:
                arr = arr[:, :dims]
            # Expand to full length if valid_mask was used
            if valid_mask is not None and len(arr) < total_frames:
                full = np.zeros((total_frames, dims), dtype=np.float32)
                full[valid_mask] = arr
                return full
            return arr
        else:
            print(f"  umap_{name}: NOT FOUND (filling with zeros)")
            return np.zeros((total_frames, dims), dtype=np.float32)

    umap_qpos = load_or_zeros('qpos', 3)
    umap_qvel = load_or_zeros('qvel', 3)
    umap_combined = load_or_zeros('combined', 3)

    com_speed_path = os.path.join(args.umap_dir, 'com_speed.npy')
    if os.path.exists(com_speed_path):
        com_speed = np.load(com_speed_path).astype(np.float32)
        print(f"  com_speed: {com_speed.shape}")
    else:
        com_speed = np.zeros(total_frames, dtype=np.float32)

    # After valid_mask expansion, all arrays should be total_frames long.
    # If no valid_mask and sizes differ, the preprocessing trimmed frames —
    # use the actual embedding size and rebuild trial_lengths from metadata.
    if len(umap_qpos) != total_frames:
        print(f"  Frame count: embeddings={len(umap_qpos)} vs index={total_frames}")
        print(f"  Rebuilding trial index from metadata...")
        meta = np.load(os.path.join(args.umap_dir, 'metadata.npz'))
        meta_trial_ids = meta['trial_id']
        # Rebuild trial starts/lengths from the actual metadata
        new_starts = []
        new_lengths = []
        new_trial_ids = []
        pos = 0
        for tid in trial_ids:
            n = int((meta_trial_ids == tid).sum())
            if n > 0:
                new_trial_ids.append(tid)
                new_starts.append(pos)
                new_lengths.append(n)
                pos += n
        trial_ids = np.array(new_trial_ids, dtype=np.int32)
        trial_starts = np.array(new_starts, dtype=np.int64)
        trial_lengths = np.array(new_lengths, dtype=np.int32)
        n_trials = len(trial_ids)
        total_frames = int(trial_lengths.sum())
        print(f"  Rebuilt: {n_trials} trials, {total_frames} frames")

    # ── Define fields ───────────────────────────────────────────
    fields = [
        ('umap_qpos',     3, 4, DTYPE_FLOAT32),
        ('umap_qvel',     3, 4, DTYPE_FLOAT32),
        ('umap_combined', 3, 4, DTYPE_FLOAT32),
        ('com_speed',     1, 4, DTYPE_FLOAT32),
    ]
    num_fields = len(fields)
    bytes_per_frame = sum(epf * esz for _, epf, esz, _ in fields)
    num_kp = 0  # not keypoint data

    print(f"\n  Fields: {num_fields}, bytes/frame: {bytes_per_frame}")

    # ── Build binary ────────────────────────────────────────────
    print(f"\nWriting: {args.output}")

    with open(args.output, 'wb') as f:
        # Header (32 bytes)
        # [0:4]   magic
        # [4:8]   version
        # [8:12]  num_trials
        # [12:16] num_fields
        # [16:20] header_size (unused, 0)
        # [20:24] fps
        # [24:26] num_keypoints
        # [26:32] padding
        # Header: 32 bytes
        # [0:4] magic, [4:8] version, [8:12] num_trials, [12:16] num_fields,
        # [16:20] header_size, [20:24] fps, [24:26] num_keypoints, [26:32] padding
        f.write(struct.pack('<IIIIII', MAGIC, FORMAT_VERSION, n_trials, num_fields, 0, 180))
        f.write(struct.pack('<H', num_kp))
        f.write(b'\x00' * 6)  # padding to 32 bytes
        assert f.tell() == 32

        # Field descriptors
        for name, epf, esz, dtype in fields:
            f.write(pack_field_desc(name, epf, esz, dtype))

        # Align to 8 bytes for index table
        pos = f.tell()
        pad = (8 - pos % 8) % 8
        f.write(b'\x00' * pad)

        # Index table (placeholder — fill after writing data)
        index_offset = f.tell()
        f.write(b'\x00' * (n_trials * INDEX_ENTRY_SIZE))

        # Per-trial data
        trial_offsets = []
        for i in range(n_trials):
            start = int(trial_starts[i])
            length = int(trial_lengths[i])

            trial_data_offset = f.tell()
            trial_offsets.append(trial_data_offset)

            # Interleave fields per frame (matching TrajReader's layout)
            for frame in range(length):
                idx = start + frame
                f.write(umap_qpos[idx].tobytes())      # 12 bytes
                f.write(umap_qvel[idx].tobytes())       # 12 bytes
                f.write(umap_combined[idx].tobytes())    # 12 bytes
                f.write(struct.pack('<f', com_speed[idx]))  # 4 bytes

            if (i + 1) % 500 == 0:
                print(f"  {i+1}/{n_trials} trials...")

        # Go back and fill index table
        f.seek(index_offset)
        for i in range(n_trials):
            f.write(struct.pack('<QI', trial_offsets[i], int(trial_lengths[i])))

    file_size = os.path.getsize(args.output)
    print(f"\nDone: {args.output} ({file_size/1e6:.1f} MB)")
    print(f"  {n_trials} trials, {total_frames} frames, {bytes_per_frame} bytes/frame")


if __name__ == "__main__":
    main()
