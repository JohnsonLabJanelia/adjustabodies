#!/usr/bin/env python3
"""Warm-started batch IK for the full Green dataset.

Processes each trial sequentially with warm-starting (frame N uses frame
N-1's qpos). Parallelizes across trials using multiprocessing. Produces
temporally consistent poses essential for UMAP embedding.

Usage:
    python3 scripts/batch_ik_warmstart.py \
        --green-dir /path/to/green \
        --model /path/to/model.mjb \
        --output /path/to/qpos_warmstart.csv \
        --metrics-csv /path/to/computed_metrics.csv \
        --workers 8
"""

import argparse
import os
import struct
import time
import sys
import numpy as np
from multiprocessing import Pool, Value


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

    # Trim to analysis window
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


# Global for multiprocessing workers
_worker_model = None
_worker_site_ids_list = None


def init_worker(model_path):
    """Initialize worker process with its own MjModel/MjData."""
    global _worker_model, _worker_site_ids_list
    import mujoco
    from adjustabodies.model import load_model, build_site_indices
    from adjustabodies.species.rodent import RAT24_SITES
    _worker_model = load_model(model_path, add_free_joint=True)
    _worker_site_ids_list = build_site_indices(_worker_model, RAT24_SITES)


def solve_trial(args):
    """Solve one trial with warm-starting. Called by Pool.map."""
    tid, frames, frame_indices = args
    from adjustabodies.ik_cpu import batch_ik_cpu_trial

    if len(frames) < 5:
        return tid, None, None, None

    qpos, residuals = batch_ik_cpu_trial(
        _worker_model, frames, _worker_site_ids_list,
        max_iters=1000, warm_iters=200, lr=0.01)

    return tid, qpos, residuals * 1000.0, frame_indices  # residuals in mm


def main():
    parser = argparse.ArgumentParser(description="Warm-started batch IK")
    parser.add_argument('--green-dir', required=True)
    parser.add_argument('--model', required=True)
    parser.add_argument('--output', default=None)
    parser.add_argument('--traj', default='green_traj3d.bin')
    parser.add_argument('--metrics-csv', default=None)
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--max-trials', type=int, default=0, help="0 = all")
    args = parser.parse_args()

    if args.output is None:
        args.output = os.path.join(args.green_dir, 'qpos_warmstart.csv')

    # Load data
    traj_path = os.path.join(args.green_dir, args.traj)
    print(f"Loading: {traj_path}")
    bindata, trials_index, traj3d = load_green_index(traj_path)
    n_trials = len(trials_index)
    print(f"  {n_trials} trials")

    # Load analysis windows
    trial_windows = {}
    metrics_csv = args.metrics_csv
    if not metrics_csv:
        candidate = os.path.join(args.green_dir, 'computed_metrics.csv')
        if os.path.exists(candidate):
            metrics_csv = candidate
    if metrics_csv and os.path.exists(metrics_csv):
        import csv
        with open(metrics_csv) as f:
            reader = csv.DictReader(f)
            for row in reader:
                tid = int(row['trial_id'])
                s = row.get('trial_start_frame', '')
                e = row.get('trial_end_frame', '')
                if s and e:
                    try:
                        trial_windows[tid] = (int(float(s)), int(float(e)))
                    except ValueError:
                        pass
        print(f"  {len(trial_windows)} trials with analysis windows")

    # Get model nq
    import mujoco
    from adjustabodies.model import load_model
    m_tmp = load_model(args.model, add_free_joint=True)
    nq = m_tmp.nq
    del m_tmp
    print(f"  Model: nq={nq}")

    # Build work items
    max_t = args.max_trials if args.max_trials > 0 else n_trials
    work = []
    total_frames = 0
    for tid in range(min(n_trials, max_t)):
        win = trial_windows.get(tid)
        ws, we = (win if win else (None, None))
        frames, frame_indices = extract_trial_frames(bindata, trials_index, traj3d, tid, ws, we)
        if len(frames) >= 5:
            work.append((tid, frames, frame_indices))
            total_frames += len(frames)

    print(f"  {len(work)} trials to process, {total_frames} frames")
    print(f"  Workers: {args.workers}")

    # Process with multiprocessing
    print(f"\nSolving IK (warm-start)...")
    t0 = time.time()

    with open(args.output, 'w') as fout:
        fout.write(f"# GREEN warm-start IK export\n")
        fout.write(f"# model: {args.model}\n")
        fout.write(f"# nq: {nq}\n")
        fout.write(f"# warm_iters: 200\n")
        fout.write(f"# cold_iters: 1000\n")
        cols = ["trial", "frame"] + [f"qpos_{i}" for i in range(nq)] + ["residual_mm"]
        fout.write(",".join(cols) + "\n")

        done = 0
        with Pool(args.workers, initializer=init_worker, initargs=(args.model,)) as pool:
            for tid, qpos, residuals, frame_indices in pool.imap(solve_trial, work):
                if qpos is None:
                    continue
                for i in range(len(qpos)):
                    parts = [str(tid), str(frame_indices[i])]
                    parts += [f"{q:.8f}" for q in qpos[i]]
                    r = residuals[i] if not np.isnan(residuals[i]) else -1.0
                    parts.append(f"{r:.4f}")
                    fout.write(",".join(parts) + "\n")
                done += len(qpos)
                elapsed = time.time() - t0
                fps = done / max(elapsed, 0.001)
                if tid % 100 == 0:
                    print(f"  Trial {tid}: {done}/{total_frames} frames  "
                          f"{fps:.0f} fps  {elapsed:.0f}s")

    elapsed = time.time() - t0
    print(f"\nDone: {done} frames in {elapsed:.0f}s ({done/max(elapsed,0.001):.0f} fps)")
    print(f"Output: {args.output}")


if __name__ == "__main__":
    main()
