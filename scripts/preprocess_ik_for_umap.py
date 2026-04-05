#!/usr/bin/env python3
"""Preprocess IK qpos export → numpy arrays for UMAP embedding.

Reads qpos_export_mjx.csv, computes qvel via mj_differentiatePos,
extracts hinge joint features, attaches metadata from Green database.

Outputs (in --output-dir):
  qpos_hinges.npy    [N, 61] float32 — hinge joint angles
  qvel_hinges.npy    [N, 61] float32 — smoothed hinge joint velocities
  com_speed.npy      [N]    float32 — COM translational speed (m/s)
  metadata.npz       trial_id, frame_idx, residual_mm, animal, session, ...
  frame_index.npz    mapping from flat index to (trial, frame) for trajectory tracing

Usage:
    python3 scripts/preprocess_ik_for_umap.py \
        --qpos-csv /path/to/qpos_export_mjx.csv \
        --model /path/to/rodent_red_mj_dev_fitted.mjb \
        --green-db /path/to/green.duckdb \
        --output-dir /path/to/umap_data \
        --fps 180 --smooth-window 37
"""

import argparse
import os
import sys
import time
import numpy as np


def parse_qpos_csv(csv_path, max_residual_mm=None):
    """Parse qpos_export_mjx.csv into per-trial arrays.

    Returns:
        trials: dict[trial_id] → {'qpos': [T, nq], 'frames': [T], 'residuals': [T]}
        nq: number of qpos columns
    """
    trials = {}
    nq = None

    print(f"Parsing: {csv_path}")
    t0 = time.time()
    n_lines = 0

    with open(csv_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                if '# nq:' in line:
                    nq = int(line.split(':')[1].strip())
                continue
            if line.startswith('trial,'):
                cols = line.split(',')
                qpos_cols = [c for c in cols if c.startswith('qpos_')]
                if qpos_cols:
                    nq = len(qpos_cols)
                continue

            parts = line.split(',')
            trial_id = int(parts[0])
            frame_idx = int(parts[1])

            if nq is None:
                # Auto-detect: trial, frame, qpos_0..nq-1, residual_mm
                nq = len(parts) - 3

            qpos = np.array([float(x) for x in parts[2:2+nq]], dtype=np.float64)
            residual = float(parts[2+nq])

            if max_residual_mm is not None and residual > max_residual_mm:
                continue

            if trial_id not in trials:
                trials[trial_id] = {'qpos': [], 'frames': [], 'residuals': []}
            trials[trial_id]['qpos'].append(qpos)
            trials[trial_id]['frames'].append(frame_idx)
            trials[trial_id]['residuals'].append(residual)

            n_lines += 1
            if n_lines % 500000 == 0:
                print(f"  {n_lines/1e6:.1f}M lines...")

    # Convert lists to arrays
    for tid in trials:
        trials[tid]['qpos'] = np.array(trials[tid]['qpos'])
        trials[tid]['frames'] = np.array(trials[tid]['frames'], dtype=np.int32)
        trials[tid]['residuals'] = np.array(trials[tid]['residuals'], dtype=np.float32)

    elapsed = time.time() - t0
    total_frames = sum(len(t['frames']) for t in trials.values())
    print(f"  {len(trials)} trials, {total_frames} frames, nq={nq} ({elapsed:.0f}s)")

    return trials, nq


def main():
    parser = argparse.ArgumentParser(description="Preprocess IK output for UMAP")
    parser.add_argument('--qpos-csv', required=True, help="qpos_export_mjx.csv path")
    parser.add_argument('--model', required=True, help="MuJoCo model (.mjb)")
    parser.add_argument('--green-db', default=None, help="green.duckdb for metadata")
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--fps', type=float, default=180.0)
    parser.add_argument('--smooth-window', type=int, default=37,
                        help="Savitzky-Golay window (frames). 0=no smoothing.")
    parser.add_argument('--max-residual', type=float, default=None,
                        help="Discard frames with residual above this (mm)")
    parser.add_argument('--metrics-csv', default=None,
                        help="computed_metrics.csv for analysis window trimming")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # ── Parse CSV ───────────────────────────────────────────────
    trials, nq = parse_qpos_csv(args.qpos_csv, args.max_residual)

    # ── Load model ──────────────────────────────────────────────
    import mujoco
    from adjustabodies.model import load_model
    from adjustabodies.qvel import compute_qvel, extract_hinge_features

    m = load_model(args.model, add_free_joint=True)
    assert m.nq == nq, f"Model nq={m.nq} != CSV nq={nq}"
    print(f"Model: nq={m.nq} nv={m.nv}")

    # ── Load metadata from database ─────────────────────────────
    trial_meta = {}
    if args.green_db and os.path.exists(args.green_db):
        import duckdb
        db = duckdb.connect(args.green_db, read_only=True)
        rows = db.sql(
            "SELECT id, animal, session_name, curr_day, is_success, dataset "
            "FROM trials ORDER BY id").fetchall()
        for row in rows:
            trial_meta[row[0]] = {
                'animal': row[1], 'session': row[2], 'curr_day': row[3],
                'is_success': row[4], 'dataset': row[5],
            }
        db.close()
        print(f"Loaded metadata for {len(trial_meta)} trials")

    # Load analysis window from computed_metrics.csv (from Green's Process All)
    trial_windows = {}
    metrics_csv = args.metrics_csv
    if not metrics_csv:
        # Try default location next to qpos CSV
        candidate = os.path.join(os.path.dirname(args.qpos_csv), 'computed_metrics.csv')
        if os.path.exists(candidate):
            metrics_csv = candidate
    if metrics_csv and os.path.exists(metrics_csv):
        import csv
        with open(metrics_csv) as f:
            reader = csv.DictReader(f)
            for row in reader:
                tid = int(row['trial_id'])
                start = row.get('trial_start_frame', '')
                end = row.get('trial_end_frame', '')
                if start and end:
                    try:
                        trial_windows[tid] = (int(float(start)), int(float(end)))
                    except ValueError:
                        pass
        print(f"Loaded analysis windows for {len(trial_windows)} trials from {metrics_csv}")
    else:
        print("WARNING: No computed_metrics.csv found — using all frames per trial")

    # ── Compute qvel per trial ──────────────────────────────────
    print(f"\nComputing qvel (fps={args.fps}, smooth={args.smooth_window})...")
    t0 = time.time()

    all_qpos_hinges = []
    all_qvel_hinges = []
    all_com_speed = []
    all_trial_ids = []
    all_frame_idxs = []
    all_residuals = []
    all_animals = []
    all_sessions = []
    all_days = []
    all_success = []
    all_datasets = []

    trial_ids_sorted = sorted(trials.keys())
    n_trimmed = 0
    for i, tid in enumerate(trial_ids_sorted):
        t = trials[tid]
        qpos_series = t['qpos']
        frames_arr = t['frames']
        residuals_arr = t['residuals']
        T = len(qpos_series)

        if T < 10:
            continue

        # Trim to analysis window (trial_start_frame → trial_end_frame)
        if tid in trial_windows:
            win_start, win_end = trial_windows[tid]
            mask = (frames_arr >= win_start) & (frames_arr <= win_end)
            if mask.sum() < 10:
                continue
            qpos_series = qpos_series[mask]
            frames_arr = frames_arr[mask]
            residuals_arr = residuals_arr[mask]
            T = len(qpos_series)
            n_trimmed += 1

        # Compute qvel for this trial (consecutive frames enable warm-start)
        qvel_series = compute_qvel(m, qpos_series, fps=args.fps,
                                    smooth_window=args.smooth_window)

        # Extract hinge features
        feats = extract_hinge_features(qpos_series, qvel_series, m)

        all_qpos_hinges.append(feats['qpos_hinges'])
        all_qvel_hinges.append(feats['qvel_hinges'])
        all_com_speed.append(feats['com_speed'])
        all_trial_ids.append(np.full(T, tid, dtype=np.int32))
        all_frame_idxs.append(frames_arr)
        all_residuals.append(residuals_arr)

        # Metadata
        meta = trial_meta.get(tid, {})
        all_animals.append(np.full(T, meta.get('animal', ''), dtype='U20'))
        all_sessions.append(np.full(T, meta.get('session', ''), dtype='U40'))
        all_days.append(np.full(T, meta.get('curr_day', -1), dtype=np.int16))
        all_success.append(np.full(T, meta.get('is_success', -1), dtype=np.int8))
        all_datasets.append(np.full(T, meta.get('dataset', ''), dtype='U20'))

        if (i + 1) % 500 == 0:
            n_so_far = sum(len(a) for a in all_qpos_hinges)
            print(f"  {i+1}/{len(trial_ids_sorted)} trials, {n_so_far/1e6:.1f}M frames...")

    print(f"  Trimmed {n_trimmed} trials to active phase (arena→return)")

    # ── Concatenate ─────────────────────────────────────────────
    print("Concatenating...")
    qpos_hinges = np.concatenate(all_qpos_hinges)
    qvel_hinges = np.concatenate(all_qvel_hinges)
    com_speed = np.concatenate(all_com_speed)
    trial_ids = np.concatenate(all_trial_ids)
    frame_idxs = np.concatenate(all_frame_idxs)
    residuals = np.concatenate(all_residuals)
    animals = np.concatenate(all_animals)
    sessions = np.concatenate(all_sessions)
    days = np.concatenate(all_days)
    success = np.concatenate(all_success)
    datasets = np.concatenate(all_datasets)

    N = len(qpos_hinges)
    elapsed = time.time() - t0
    print(f"Done: {N} frames from {len(trial_ids_sorted)} trials ({elapsed:.0f}s)")
    print(f"  qpos_hinges: {qpos_hinges.shape}")
    print(f"  qvel_hinges: {qvel_hinges.shape}")

    # ── Save ────────────────────────────────────────────────────
    print(f"\nSaving to {args.output_dir}/")

    np.save(os.path.join(args.output_dir, 'qpos_hinges.npy'), qpos_hinges)
    np.save(os.path.join(args.output_dir, 'qvel_hinges.npy'), qvel_hinges)
    np.save(os.path.join(args.output_dir, 'com_speed.npy'), com_speed)

    np.savez_compressed(os.path.join(args.output_dir, 'metadata.npz'),
        trial_id=trial_ids, frame_idx=frame_idxs, residual_mm=residuals,
        animal=animals, session=sessions, curr_day=days,
        is_success=success, dataset=datasets)

    # Trial boundaries for trajectory tracing
    trial_starts = []
    trial_lengths = []
    pos = 0
    for tid in trial_ids_sorted:
        t = trials[tid]
        T = len(t['frames'])
        if T < 10:
            continue
        trial_starts.append(pos)
        trial_lengths.append(T)
        pos += T
    np.savez_compressed(os.path.join(args.output_dir, 'trial_index.npz'),
        trial_ids=np.array(trial_ids_sorted, dtype=np.int32),
        starts=np.array(trial_starts, dtype=np.int64),
        lengths=np.array(trial_lengths, dtype=np.int32))

    sizes = {f: os.path.getsize(os.path.join(args.output_dir, f)) / 1e6
             for f in os.listdir(args.output_dir)}
    for f, sz in sorted(sizes.items()):
        print(f"  {f}: {sz:.1f} MB")

    print(f"\nTotal: {sum(sizes.values()):.0f} MB")
    print("Done.")


if __name__ == "__main__":
    main()
