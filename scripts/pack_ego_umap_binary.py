#!/usr/bin/env python3
"""Pack egocentric UMAP outputs into Green's TrajReader binary format.

Reads the UMAP embedding and metadata from preprocess_ego_umap.py + gpu_umap_ego.py,
packs into green_umap.bin for the UMAP panel.

Output fields per frame:
    umap_ego       3 × float32   Egocentric UMAP embedding (primary)
    umap_qpos      3 × float32   (copy of ego — backwards compat)
    umap_qvel      3 × float32   (copy of ego — backwards compat)
    umap_combined  3 × float32   (copy of ego — backwards compat)
    com_speed      1 × float32   COM translational speed (mm/frame)

Usage:
    python3 scripts/pack_ego_umap_binary.py \
        --umap-dir /path/to/umap_ego \
        --output /path/to/green_umap.bin
"""

import argparse
import os
import struct
import numpy as np


MAGIC = 0x024E5247
FORMAT_VERSION = 2
FIELD_NAME_LEN = 32
FIELD_DESC_SIZE = 44
INDEX_ENTRY_SIZE = 12
DTYPE_FLOAT32 = 0


def pack_field_desc(name, elements_per_frame, element_size, dtype_code):
    """Pack a 44-byte field descriptor."""
    name_bytes = name.encode('utf-8')[:FIELD_NAME_LEN]
    name_bytes = name_bytes + b'\x00' * (FIELD_NAME_LEN - len(name_bytes))
    return name_bytes + struct.pack('<III', elements_per_frame, element_size, dtype_code)


def main():
    parser = argparse.ArgumentParser(description="Pack egocentric UMAP into Green binary")
    parser.add_argument('--umap-dir', required=True, help="Directory with ego UMAP outputs")
    parser.add_argument('--output', required=True, help="Output .bin path")
    args = parser.parse_args()

    print(f"Loading from {args.umap_dir}/")

    # Trial index
    tidx = np.load(os.path.join(args.umap_dir, 'trial_index.npz'))
    trial_ids = tidx['trial_ids']
    trial_starts = tidx['starts']
    trial_lengths = tidx['lengths']
    n_trials = len(trial_ids)
    total_frames = int(trial_lengths.sum())
    print(f"  {n_trials} trials, {total_frames} total frames")

    # Load UMAP embedding
    umap_path = os.path.join(args.umap_dir, 'umap_ego.npy')
    if os.path.exists(umap_path):
        umap_ego = np.load(umap_path).astype(np.float32)
        print(f"  umap_ego: {umap_ego.shape}")
        # Pad to 3D if needed
        if umap_ego.shape[1] < 3:
            umap_ego = np.concatenate([umap_ego,
                np.zeros((umap_ego.shape[0], 3 - umap_ego.shape[1]), dtype=np.float32)], axis=1)
        elif umap_ego.shape[1] > 3:
            umap_ego = umap_ego[:, :3]
    else:
        print("  WARNING: umap_ego.npy not found — filling with zeros")
        umap_ego = np.zeros((total_frames, 3), dtype=np.float32)

    # Replace NaN with 0 for display
    umap_ego = np.nan_to_num(umap_ego, nan=0.0)

    # COM speed
    com_speed = np.load(os.path.join(args.umap_dir, 'com_speed.npy')).astype(np.float32)
    print(f"  com_speed: {com_speed.shape}")

    # Animal IDs (optional — for per-animal coloring)
    animal_path = os.path.join(args.umap_dir, 'animal_ids.npy')
    has_animals = os.path.exists(animal_path)
    if has_animals:
        animal_ids = np.load(animal_path).astype(np.uint8)
        print(f"  animal_ids: {animal_ids.shape}")

    # Build per-frame trial IDs and frame numbers for cursor mapping
    # trial_id: database trial ID (constant within a trial)
    # frame_id: global frame number within the trial
    trial_id_per_frame = np.zeros(total_frames, dtype=np.int32)
    frame_ids = np.zeros(total_frames, dtype=np.int32)

    frame_ids_path = os.path.join(args.umap_dir, 'frame_ids.npy')
    if os.path.exists(frame_ids_path):
        frame_ids = np.load(frame_ids_path).astype(np.int32)
        print(f"  frame_ids: {frame_ids.shape}")
    else:
        for i in range(n_trials):
            s, l = int(trial_starts[i]), int(trial_lengths[i])
            frame_ids[s:s+l] = np.arange(l, dtype=np.int32)
        print(f"  frame_ids: sequential fallback")

    # Fill trial_id per frame from trial_index
    for i in range(n_trials):
        s, l = int(trial_starts[i]), int(trial_lengths[i])
        trial_id_per_frame[s:s+l] = trial_ids[i]
    print(f"  trial_ids: {n_trials} trials, range [{trial_ids.min()}, {trial_ids.max()}]")

    # Verify sizes
    assert len(umap_ego) == total_frames, f"umap_ego {len(umap_ego)} != {total_frames}"
    assert len(com_speed) == total_frames, f"com_speed {len(com_speed)} != {total_frames}"

    # ── Fields ──────────────────────────────────────────────────
    DTYPE_INT32 = 2
    fields = [
        ('umap_ego',      3, 4, DTYPE_FLOAT32),
        ('umap_qpos',     3, 4, DTYPE_FLOAT32),  # copy of ego for compat
        ('umap_qvel',     3, 4, DTYPE_FLOAT32),  # copy of ego for compat
        ('umap_combined', 3, 4, DTYPE_FLOAT32),  # copy of ego for compat
        ('com_speed',     1, 4, DTYPE_FLOAT32),
        ('trial_id',      1, 4, DTYPE_INT32),     # database trial ID
        ('frame_id',      1, 4, DTYPE_INT32),     # global frame number within trial
    ]
    if has_animals:
        fields.append(('animal_id', 1, 1, 1))  # dtype_code 1 = uint8
    num_fields = len(fields)
    bytes_per_frame = sum(epf * esz for _, epf, esz, _ in fields)

    print(f"\n  Fields: {num_fields}, bytes/frame: {bytes_per_frame}")

    # ── Write binary ────────────────────────────────────────────
    print(f"\nWriting: {args.output}")

    with open(args.output, 'wb') as f:
        # Header (32 bytes)
        f.write(struct.pack('<IIIIII', MAGIC, FORMAT_VERSION, n_trials, num_fields, 0, 180))
        f.write(struct.pack('<H', 0))  # num_keypoints (not keypoint data)
        f.write(b'\x00' * 6)
        assert f.tell() == 32

        # Field descriptors
        for name, epf, esz, dtype in fields:
            f.write(pack_field_desc(name, epf, esz, dtype))

        # Align to 8 bytes
        pos = f.tell()
        pad = (8 - pos % 8) % 8
        f.write(b'\x00' * pad)

        # Index table (placeholder)
        index_offset = f.tell()
        f.write(b'\x00' * (n_trials * INDEX_ENTRY_SIZE))

        # Per-trial data: contiguous field blocks (NOT interleaved per frame)
        # Layout: [all frames of field 0][all frames of field 1]...
        # This matches TrajReader's access: offset + num_frames * byte_offset_per_frame
        trial_offsets = []
        for i in range(n_trials):
            start = int(trial_starts[i])
            length = int(trial_lengths[i])
            end = start + length

            trial_offsets.append(f.tell())

            # Field 0: umap_ego (3 × float32)
            f.write(umap_ego[start:end].tobytes())
            # Field 1: umap_qpos (copy of ego)
            f.write(umap_ego[start:end].tobytes())
            # Field 2: umap_qvel (copy of ego)
            f.write(umap_ego[start:end].tobytes())
            # Field 3: umap_combined (copy of ego)
            f.write(umap_ego[start:end].tobytes())
            # Field 4: com_speed (1 × float32)
            f.write(com_speed[start:end].tobytes())
            # Field 5: trial_id (1 × int32)
            f.write(trial_id_per_frame[start:end].tobytes())
            # Field 6: frame_id (1 × int32)
            f.write(frame_ids[start:end].tobytes())
            # Field 7: animal_id (1 × uint8, optional)
            if has_animals:
                f.write(animal_ids[start:end].tobytes())

            if (i + 1) % 500 == 0:
                print(f"  {i+1}/{n_trials} trials...")

        # Fill index table
        f.seek(index_offset)
        for i in range(n_trials):
            f.write(struct.pack('<QI', trial_offsets[i], int(trial_lengths[i])))

    file_size = os.path.getsize(args.output)
    print(f"\nDone: {args.output} ({file_size/1e6:.1f} MB)")
    print(f"  {n_trials} trials, {total_frames} frames, {bytes_per_frame} bytes/frame")


if __name__ == "__main__":
    main()
