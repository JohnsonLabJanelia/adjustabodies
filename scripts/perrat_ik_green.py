#!/usr/bin/env python3
"""Per-rat IK evaluation on Green dataset using MJX GPU solver.

Extracts 500 random poses per rat from green_traj3d.bin, runs MJX IK,
and reports per-rat residual statistics. This reveals how well the
calibrated body model fits each of the 5 Green rats.

Usage (cluster GPU):
    python3 scripts/perrat_ik_green.py \
        --green-dir /groups/johnson/johnsonlab/virtual_rodent/green \
        --green-db /groups/johnson/johnsonlab/virtual_rodent/green/green.duckdb \
        --model /groups/johnson/johnsonlab/virtual_rodent/body_model/rodent_red_mj_dev_fitted.mjb \
        --frames-per-rat 500
"""

import argparse
import os
import sys
import struct
import time
import numpy as np


def load_green_binary(path):
    """Load green_traj3d.bin index and field info."""
    data = np.memmap(path, dtype=np.uint8, mode='r')
    magic = struct.unpack_from('<I', data, 0)[0]
    assert magic == 0x024E5247, f"Bad magic: {magic:#x}"

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

    # Parse trial index
    trials_index = []
    for i in range(num_trials):
        pos = index_start + i * 12
        offset = struct.unpack_from('<Q', data, pos)[0]
        nf = struct.unpack_from('<I', data, pos + 8)[0]
        trials_index.append((offset, nf))

    return data, trials_index, traj3d


def extract_random_frames(data, trials_index, traj3d, trial_ids, n_frames, rng):
    """Extract n_frames random frames from the given trials.

    Returns list of (kp3d[24,3], valid[24]) tuples in meters.
    """
    stride = traj3d['epf']
    traj_offset = traj3d['offset']

    # Build pool of (trial_id, frame_idx) pairs
    pool = []
    for tid in trial_ids:
        offset, nf = trials_index[tid]
        # Skip first/last 10% of each trial (start/stop artifacts)
        start_f = max(0, int(nf * 0.1))
        end_f = min(nf, int(nf * 0.9))
        for f in range(start_f, end_f):
            pool.append((tid, f, offset, nf))

    if len(pool) == 0:
        return []

    # Random sample
    n_sample = min(n_frames, len(pool))
    indices = rng.choice(len(pool), size=n_sample, replace=False)

    frames = []
    for idx in indices:
        tid, fi, trial_offset, nf = pool[idx]
        start = trial_offset + nf * traj_offset
        # Read single frame
        frame_start = start + fi * stride * 4
        arr = np.frombuffer(data[frame_start:frame_start + stride * 4], dtype=np.float32)
        kp = arr[:72].reshape(24, 3).copy() * 0.001  # mm → meters
        valid = np.isfinite(kp).all(axis=-1) & ~(kp == 0).all(axis=-1)
        frames.append((kp.astype(np.float32), valid.astype(np.float32)))

    return frames


def main():
    parser = argparse.ArgumentParser(description="Per-rat IK evaluation on Green data")
    parser.add_argument('--green-dir', required=True, help="Green data directory")
    parser.add_argument('--green-db', default=None, help="green.duckdb path (default: <green-dir>/green.duckdb)")
    parser.add_argument('--model', required=True, help="Fitted .mjb model")
    parser.add_argument('--traj', default='green_traj3d.bin', help="Trajectory file")
    parser.add_argument('--frames-per-rat', type=int, default=500)
    parser.add_argument('--ik-iters', type=int, default=1000)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--batch-size', type=int, default=512)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    if args.green_db is None:
        args.green_db = os.path.join(args.green_dir, 'green.duckdb')

    # ── Load Green binary ───────────────────────────────────────
    traj_path = os.path.join(args.green_dir, args.traj)
    print(f"Loading: {traj_path}")
    bindata, trials_index, traj3d = load_green_binary(traj_path)
    print(f"  {len(trials_index)} trials")

    # ── Load database for rat→trial mapping ─────────────────────
    import duckdb
    db = duckdb.connect(args.green_db, read_only=True)
    rats = [r[0] for r in db.sql("SELECT DISTINCT animal FROM trials ORDER BY animal").fetchall()]
    print(f"  Rats: {rats}")

    rat_trials = {}
    for rat in rats:
        tids = [r[0] for r in db.sql(f"SELECT id FROM trials WHERE animal='{rat}' ORDER BY id").fetchall()]
        rat_trials[rat] = tids
        total_frames = sum(trials_index[t][1] for t in tids)
        print(f"    {rat}: {len(tids)} trials, ~{total_frames} frames")

    # ── Load model ──────────────────────────────────────────────
    from adjustabodies.model import load_model, build_site_indices
    from adjustabodies.species.rodent import RAT24_SITES

    print(f"\nLoading model: {args.model}")
    m = load_model(args.model, add_free_joint=True, fix_geoms_for_mjx=True)
    site_ids = build_site_indices(m, RAT24_SITES)
    mapped = sum(1 for s in site_ids if s >= 0)
    print(f"  nq={m.nq} nv={m.nv} sites={mapped}/24")

    # ── Build MJX solver ────────────────────────────────────────
    import jax
    from adjustabodies.ik_mjx import build_ik_solver, IKConfig
    import jax.numpy as jnp

    has_gpu = any(d.platform == 'gpu' for d in jax.devices())
    print(f"\nJAX devices: {jax.devices()}")

    config = IKConfig(
        max_iters=args.ik_iters,
        lr=args.lr,
        batch_size=args.batch_size,
        use_scan=has_gpu,
    )
    print(f"Building solver ({'scan' if config.use_scan else 'step'} mode, "
          f"iters={config.max_iters}, lr={config.lr})...")

    t0 = time.time()
    solve_batch, nq = build_ik_solver(m, site_ids, config)

    # Warmup compilation
    dummy = jnp.zeros((args.batch_size, 24, 3))
    dummy_v = jnp.ones((args.batch_size, 24))
    _ = solve_batch(dummy, dummy_v)
    jax.block_until_ready(_[0])
    print(f"Compiled in {time.time() - t0:.1f}s")

    # ── Run IK per rat ──────────────────────────────────────────
    rng = np.random.default_rng(args.seed)
    bs = args.batch_size

    print(f"\n{'='*70}")
    print(f"{'Rat':12s} {'N':>5s} {'Mean':>8s} {'Median':>8s} {'p5':>8s} "
          f"{'p95':>8s} {'Max':>8s} {'Time':>6s}")
    print(f"{'='*70}")

    all_results = {}

    for rat in rats:
        # Extract random frames
        frames = extract_random_frames(
            bindata, trials_index, traj3d, rat_trials[rat],
            args.frames_per_rat, rng)

        if not frames:
            print(f"{rat:12s} — no valid frames")
            continue

        N = len(frames)
        kp3d_all = np.stack([f[0] for f in frames])
        valid_all = np.stack([f[1] for f in frames])

        # Solve in batches
        all_qpos = []
        all_res = []
        t_solve = time.time()

        for b_start in range(0, N, bs):
            b_end = min(b_start + bs, N)
            actual = b_end - b_start

            kp_batch = kp3d_all[b_start:b_end]
            v_batch = valid_all[b_start:b_end]

            if actual < bs:
                kp_batch = np.concatenate([kp_batch, np.zeros((bs - actual, 24, 3), dtype=np.float32)])
                v_batch = np.concatenate([v_batch, np.zeros((bs - actual, 24), dtype=np.float32)])

            qpos_b, res_b = solve_batch(jnp.array(kp_batch), jnp.array(v_batch))
            jax.block_until_ready(qpos_b)

            all_res.append(np.array(res_b[:actual]) * 1000.0)

        elapsed = time.time() - t_solve
        residuals = np.concatenate(all_res)

        # Filter valid (residual > 0, not NaN)
        valid_mask = (residuals > 0) & np.isfinite(residuals)
        r = residuals[valid_mask]

        if len(r) > 0:
            print(f"{rat:12s} {len(r):5d} {r.mean():8.2f} {np.median(r):8.2f} "
                  f"{np.percentile(r, 5):8.2f} {np.percentile(r, 95):8.2f} "
                  f"{r.max():8.2f} {elapsed:5.1f}s")
            all_results[rat] = r
        else:
            print(f"{rat:12s} — all frames failed")

    # ── Summary ─────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("Summary")
    print(f"{'='*70}")

    if all_results:
        all_r = np.concatenate(list(all_results.values()))
        print(f"{'ALL':12s} {len(all_r):5d} {all_r.mean():8.2f} {np.median(all_r):8.2f} "
              f"{np.percentile(all_r, 5):8.2f} {np.percentile(all_r, 95):8.2f} "
              f"{all_r.max():8.2f}")

        # Rank rats by median residual
        print("\nRanked by median residual (best → worst):")
        ranked = sorted(all_results.items(), key=lambda x: np.median(x[1]))
        for rat, r in ranked:
            print(f"  {rat:12s}: median={np.median(r):.2f}mm  "
                  f"(body model fits {'well' if np.median(r) < 7 else 'moderately' if np.median(r) < 12 else 'poorly'})")

    db.close()
    print("\nDone.")


if __name__ == "__main__":
    main()
