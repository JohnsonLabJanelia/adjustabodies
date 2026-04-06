#!/usr/bin/env python3
"""Batch IK with per-animal v4 fitted models.

Uses the correct fitted body model for each rat, with warm-starting
across frames within each trial. Outputs qpos + residuals as CSV.

Usage:
    python3 scripts/batch_ik_v4.py \
        --green-dir /path/to/green \
        --models-dir /path/to/green_fits_v4 \
        --output /path/to/qpos_v4.csv \
        --workers 8
"""

import argparse
import csv
import os
import struct
import sys
import time
import numpy as np
from multiprocessing import Pool

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


def load_green_index(traj_path):
    """Load trial index from green binary."""
    data = np.memmap(traj_path, dtype=np.uint8, mode='r')
    num_trials = struct.unpack_from('<I', data, 8)[0]
    num_fields = struct.unpack_from('<I', data, 12)[0]

    fields = []
    cumulative = 0
    for i in range(num_fields):
        pos = 32 + i * 44
        name = bytes(data[pos:pos+32]).split(b'\0')[0].decode()
        epf = struct.unpack_from('<I', data, pos + 32)[0]
        esz = struct.unpack_from('<I', data, pos + 36)[0]
        fields.append({'name': name, 'epf': epf, 'esz': esz, 'offset': cumulative})
        cumulative += epf * esz

    traj3d = next(f for f in fields if f['name'] == 'traj3d')
    desc_end = 32 + num_fields * 44
    index_start = (desc_end + 7) & ~7

    trials_index = []
    for i in range(num_trials):
        pos = index_start + i * 12
        offset = struct.unpack_from('<Q', data, pos)[0]
        nf = struct.unpack_from('<I', data, pos + 8)[0]
        trials_index.append((offset, nf))

    return data, trials_index, traj3d


def extract_trial_frames(data, trials_index, traj3d, tid, win_start=None, win_end=None):
    """Extract frames for one trial, optionally trimmed to analysis window."""
    offset, nf = trials_index[tid]
    stride = traj3d['epf']
    traj_offset = traj3d['offset']

    start = offset + nf * traj_offset
    nbytes = nf * stride * 4
    arr = np.frombuffer(data[start:start + nbytes], dtype=np.float32).reshape(nf, stride)
    kp3d = arr[:, :72].reshape(nf, 24, 3)

    frame_indices = np.arange(nf)
    if win_start is not None and win_end is not None:
        mask = (frame_indices >= win_start) & (frame_indices <= win_end)
        kp3d = kp3d[mask]
        frame_indices = frame_indices[mask]

    frames = []
    for i in range(len(kp3d)):
        kp = kp3d[i].copy() * 0.001  # mm → m
        valid = np.isfinite(kp).all(axis=-1) & ~(kp == 0).all(axis=-1)
        frames.append((kp.astype(np.float32), valid.astype(np.float32)))

    return frames, frame_indices


# ── Worker state ──────────────────────────────────────────────────────

_worker_models = {}  # animal_name → (MjModel, site_ids)
_worker_fallback = None  # 'all' model


def init_worker(models_dir):
    """Initialize worker with all per-animal models."""
    global _worker_models, _worker_fallback
    import mujoco
    from adjustabodies.model import load_model, build_site_indices
    from adjustabodies.species.rodent import RAT24_SITES

    animals = ['captain', 'emilie', 'heisenberg', 'mario', 'remy']
    for animal in animals:
        path = os.path.join(models_dir, f'rodent_green_{animal}.mjb')
        if os.path.exists(path):
            m = load_model(path, add_free_joint=True)
            sids = build_site_indices(m, RAT24_SITES)
            _worker_models[animal] = (m, sids)

    # Fallback: average model
    all_path = os.path.join(models_dir, 'rodent_green_all.mjb')
    if os.path.exists(all_path):
        m = load_model(all_path, add_free_joint=True)
        sids = build_site_indices(m, RAT24_SITES)
        _worker_models['all'] = (m, sids)
        _worker_fallback = 'all'


def solve_trial(args):
    """Solve one trial with the correct per-animal model."""
    tid, animal, frames, frame_indices = args
    from adjustabodies.ik_cpu import batch_ik_cpu_trial

    if len(frames) < 5:
        return tid, None, None, None, animal

    # Select model for this animal
    key = animal if animal in _worker_models else _worker_fallback
    if key is None:
        return tid, None, None, None, animal

    m, site_ids = _worker_models[key]
    qpos, residuals = batch_ik_cpu_trial(
        m, frames, site_ids,
        max_iters=1000, warm_iters=200, lr=0.01)

    return tid, qpos, residuals * 1000.0, frame_indices, animal


def main():
    parser = argparse.ArgumentParser(description="Batch IK with per-animal v4 models")
    parser.add_argument('--green-dir', required=True)
    parser.add_argument('--models-dir', required=True,
                        help="Directory with rodent_green_{animal}.mjb files")
    parser.add_argument('--output', default=None)
    parser.add_argument('--traj', default='repaired_traj3d.bin')
    parser.add_argument('--metrics-csv', default=None)
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--max-trials', type=int, default=0)
    args = parser.parse_args()

    if args.output is None:
        args.output = os.path.join(args.green_dir, 'qpos_v4.csv')

    # Load trajectory data
    traj_path = os.path.join(args.green_dir, args.traj)
    print(f"Loading: {traj_path}")
    bindata, trials_index, traj3d = load_green_index(traj_path)
    n_trials = len(trials_index)
    print(f"  {n_trials} trials")

    # Load analysis windows from computed_metrics.csv
    trial_windows = {}
    metrics_csv = args.metrics_csv or os.path.join(args.green_dir, 'computed_metrics.csv')
    if os.path.exists(metrics_csv):
        with open(metrics_csv) as f:
            reader = csv.DictReader(f)
            for row in reader:
                tid = int(row['trial_id'])
                s, e = row.get('trial_start_frame', ''), row.get('trial_end_frame', '')
                if s and e:
                    try:
                        trial_windows[tid] = (int(float(s)), int(float(e)))
                    except ValueError:
                        pass
        print(f"  {len(trial_windows)} trials with analysis windows")

    # Load animal mapping from DuckDB
    import duckdb
    db_path = os.path.join(args.green_dir, 'green.duckdb')
    con = duckdb.connect(db_path, read_only=True)
    animal_rows = con.execute("SELECT id, animal FROM trials ORDER BY id").fetchall()
    trial_animal = {r[0]: r[1] for r in animal_rows}
    con.close()
    print(f"  Animal mapping: {len(trial_animal)} trials")

    # Check available models
    print(f"\nModels directory: {args.models_dir}")
    for f in sorted(os.listdir(args.models_dir)):
        if f.endswith('.mjb'):
            print(f"  {f}")

    # Get nq from any model
    import mujoco
    from adjustabodies.model import load_model
    sample_mjb = next(os.path.join(args.models_dir, f)
                      for f in os.listdir(args.models_dir) if f.endswith('.mjb'))
    m_tmp = load_model(sample_mjb, add_free_joint=True)
    nq = m_tmp.nq
    del m_tmp
    print(f"  nq={nq}")

    # Build work items
    max_t = args.max_trials if args.max_trials > 0 else n_trials
    work = []
    total_frames = 0
    animal_counts = {}
    for tid in range(min(n_trials, max_t)):
        win = trial_windows.get(tid)
        ws, we = (win if win else (None, None))
        frames, frame_indices = extract_trial_frames(bindata, trials_index, traj3d, tid, ws, we)
        if len(frames) >= 5:
            animal = trial_animal.get(tid, 'unknown')
            work.append((tid, animal, frames, frame_indices))
            total_frames += len(frames)
            animal_counts[animal] = animal_counts.get(animal, 0) + 1

    print(f"\n  {len(work)} trials to process, {total_frames} frames")
    for animal, count in sorted(animal_counts.items()):
        print(f"    {animal}: {count} trials")
    print(f"  Workers: {args.workers}")

    # Process
    print(f"\nSolving IK (warm-start, per-animal models)...")
    t0 = time.time()

    with open(args.output, 'w') as fout:
        fout.write(f"# GREEN v4 per-animal IK export\n")
        fout.write(f"# models: {args.models_dir}\n")
        fout.write(f"# nq: {nq}\n")
        fout.write(f"# warm_iters: 200, cold_iters: 1000\n")
        cols = ["trial", "frame", "animal"] + [f"qpos_{i}" for i in range(nq)] + ["residual_mm"]
        fout.write(",".join(cols) + "\n")

        done = 0
        with Pool(args.workers, initializer=init_worker, initargs=(args.models_dir,)) as pool:
            for tid, qpos, residuals, frame_indices, animal in pool.imap(solve_trial, work):
                if qpos is None:
                    continue
                for i in range(len(qpos)):
                    parts = [str(tid), str(frame_indices[i]), animal]
                    parts += [f"{q:.8f}" for q in qpos[i]]
                    r = residuals[i] if not np.isnan(residuals[i]) else -1.0
                    parts.append(f"{r:.4f}")
                    fout.write(",".join(parts) + "\n")
                done += len(qpos)
                elapsed = time.time() - t0
                fps = done / max(elapsed, 0.001)
                if tid % 100 == 0:
                    print(f"  Trial {tid} ({animal}): {done}/{total_frames} frames  "
                          f"{fps:.0f} fps  {elapsed:.0f}s")

    elapsed = time.time() - t0
    print(f"\nDone: {done} frames in {elapsed:.0f}s ({done/max(elapsed,0.001):.0f} fps)")
    print(f"Output: {args.output}")


if __name__ == "__main__":
    main()
