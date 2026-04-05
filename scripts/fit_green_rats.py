#!/usr/bin/env python3
"""Fit body models to Green dataset: 1 average + 5 per-rat.

Extracts random poses from green_traj3d.bin, runs the adjustabodies
2-phase fitting pipeline (segment scaling + STAC site offsets) on GPU.

Produces 6 fitted .mjb models:
  - rodent_green_all.mjb        (average across all 5 rats, 2500 frames)
  - rodent_green_captain.mjb    (2500 frames from captain)
  - rodent_green_emilie.mjb     (etc.)
  - rodent_green_heisenberg.mjb
  - rodent_green_mario.mjb
  - rodent_green_remy.mjb

Usage (cluster GPU):
    python3 scripts/fit_green_rats.py \
        --green-dir /groups/johnson/johnsonlab/virtual_rodent/green \
        --green-db /groups/johnson/johnsonlab/virtual_rodent/green/green.duckdb \
        --base-model /groups/johnson/johnsonlab/virtual_rodent/body_model/rodent_data_driven_limits.xml \
        --output-dir /groups/johnson/johnsonlab/virtual_rodent/body_model/green_fits
"""

import argparse
import os
import struct
import time
import numpy as np


def load_green_binary(path):
    """Load green_traj3d.bin index and field info."""
    data = np.memmap(path, dtype=np.uint8, mode='r')
    magic = struct.unpack_from('<I', data, 0)[0]
    assert magic == 0x024E5247

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


def extract_frames(data, trials_index, traj3d, trial_ids, n_frames, rng,
                    trial_phases=None):
    """Extract n_frames random frames from active phases of given trials.

    Args:
        trial_phases: optional dict[trial_id] → (start_frame, end_frame) for active phase.
                      If provided, only samples from within these ranges.
                      If None, uses middle 80% of each trial.
    """
    stride = traj3d['epf']
    traj_offset = traj3d['offset']

    pool = []
    for tid in trial_ids:
        offset, nf = trials_index[tid]
        if trial_phases and tid in trial_phases:
            start_f, end_f = trial_phases[tid]
            start_f = max(0, int(start_f))
            end_f = min(nf, int(end_f))
        else:
            start_f = max(0, int(nf * 0.1))
            end_f = min(nf, int(nf * 0.9))
        if end_f <= start_f:
            continue
        for f in range(start_f, end_f):
            pool.append((tid, f, offset, nf))

    if not pool:
        return []

    # Oversample 3x to compensate for frames with too few valid keypoints
    n_sample = min(n_frames * 3, len(pool))
    indices = rng.choice(len(pool), size=n_sample, replace=False)

    frames = []
    for idx in indices:
        if len(frames) >= n_frames:
            break
        tid, fi, trial_offset, nf = pool[idx]
        start = trial_offset + nf * traj_offset
        frame_start = start + fi * stride * 4
        arr = np.frombuffer(data[frame_start:frame_start + stride * 4], dtype=np.float32)
        kp = arr[:72].reshape(24, 3).copy() * 0.001  # mm → m
        valid = np.isfinite(kp).all(axis=-1) & ~(kp == 0).all(axis=-1)
        # Require at least 12 valid keypoints (half the skeleton)
        if valid.sum() < 12:
            continue
        frames.append((kp.astype(np.float32), valid.astype(np.float32)))

    return frames


def fit_on_frames(model_xml, frames, output_path, label, n_rounds=6, m_iters=300, ik_iters=1000):
    """Run the 2-phase fitting pipeline on a set of frames.

    This is fit_body_model() but accepts frames directly instead of loading from CSV.
    """
    from adjustabodies import enable_jax_cache
    enable_jax_cache()
    import jax
    import jax.numpy as jnp
    import mujoco
    from mujoco import mjx
    from adjustabodies.model import (load_model, build_segment_indices, build_site_indices,
                                      save_originals, apply_segment_scales)
    from adjustabodies.io import save_fitted_model
    from adjustabodies.resize import run_resize_phase, build_mjx_scale_fn
    from adjustabodies.stac import run_stac_phase
    from adjustabodies.species.rodent import (SEGMENT_DEFS, RAT24_SITES, SEGMENT_LENGTH_INIT,
                                               LR_SITE_PAIRS, MIDLINE_SITES)

    print(f"\n{'#'*70}")
    print(f"# FITTING: {label} ({len(frames)} frames)")
    print(f"# Output:  {output_path}")
    print(f"{'#'*70}")

    t0 = time.time()

    m = load_model(model_xml, add_free_joint=True, fix_geoms_for_mjx=True)
    segments = build_segment_indices(m, SEGMENT_DEFS)
    site_ids = build_site_indices(m, RAT24_SITES)
    orig = save_originals(m)
    n_seg = len(segments)

    mx_base = mjx.put_model(m)
    apply_scales_fn = build_mjx_scale_fn(m, segments, orig)

    init_rel_scales = np.array(
        [SEGMENT_LENGTH_INIT.get(name, 1.0) for name, _ in segments],
        dtype=np.float32)

    # Phase 1: Segment scaling
    print(f"\n--- Phase 1: Segment scaling ---")
    params, pre_res, post_res_1 = run_resize_phase(
        m, mx_base, segments, site_ids, orig, frames, apply_scales_fn,
        init_global=1.0, init_rel_scales=init_rel_scales,
        n_rounds=n_rounds, m_iters=m_iters, ik_iters=ik_iters,
        lr_scale=0.003, reg_scale=0.001, verbose=True)

    gs = float(params['global_scale'])
    rs = np.array(params['rel_scales'])
    print(f"\nPhase 1 scales:")
    for g, (name, _) in enumerate(segments):
        print(f"  {name:<12s} {gs*rs[g]:.3f}")

    # Apply Phase 1 scales and prepare for Phase 2
    scales = {name: gs * rs[g] for g, (name, _) in enumerate(segments)}
    apply_segment_scales(m, segments, scales, orig)
    orig_scaled = save_originals(m)
    mx_base_scaled = mjx.put_model(m)
    apply_scales_fn_2 = build_mjx_scale_fn(m, segments, orig_scaled)

    params['rel_scales'] = jnp.ones(n_seg)
    params['global_scale'] = jnp.array(1.0)

    # Phase 2: STAC site offsets
    print(f"\n--- Phase 2: STAC site offsets ---")
    params, pre_res_2, post_res_2 = run_stac_phase(
        m, mx_base_scaled, segments, site_ids, orig_scaled, frames,
        apply_scales_fn_2, params,
        sym_config=(LR_SITE_PAIRS, MIDLINE_SITES),
        n_rounds=n_rounds, m_iters=m_iters, ik_iters=ik_iters,
        lr_scale=0.0001, lr_offset=0.001,
        reg_scale=10.0, reg_offset=0.01, verbose=True)

    # Apply final state
    gs_final = float(params['global_scale'])
    rs_final = np.array(params['rel_scales'])
    offs_final = np.array(params['site_offsets'])
    final_scales = {name: gs_final * rs_final[g] for g, (name, _) in enumerate(segments)}
    apply_segment_scales(m, segments, final_scales, orig_scaled)
    m.site_pos[:] += offs_final  # applies to all sites; non-keypoint offsets ~0 due to regularization
    mujoco.mj_setConst(m, mujoco.MjData(m))

    abs_scales = {name: scales[name] * final_scales[name] for name in scales}

    metadata = {
        'adjustabodies_version': '0.1.0',
        'base_model': model_xml,
        'data_source': f'green/{label}',
        'n_frames': len(frames),
        'phase1_residual_mm': post_res_1,
        'phase2_residual_mm': post_res_2,
        'segment_scales': abs_scales,
    }

    save_fitted_model(m, output_path, metadata)
    elapsed = time.time() - t0

    print(f"\n{'='*50}")
    print(f"RESULT: {label}")
    print(f"  Residual: {pre_res:.2f} → {post_res_1:.2f} → {post_res_2:.2f} mm")
    print(f"  Time: {elapsed:.0f}s")
    print(f"  Saved: {output_path}")
    for name, s in abs_scales.items():
        print(f"  {name:<12s} {s:.3f}")

    return metadata


def main():
    parser = argparse.ArgumentParser(description="Fit body models to Green rats")
    parser.add_argument('--green-dir', required=True)
    parser.add_argument('--green-db', default=None)
    parser.add_argument('--base-model', required=True, help="Base XML model (not fitted)")
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--traj', default='repaired_traj3d.bin')
    parser.add_argument('--frames-per-rat', type=int, default=2500)
    parser.add_argument('--frames-average', type=int, default=500,
                        help="Frames per rat for the average model (total = 5×this)")
    parser.add_argument('--n-rounds', type=int, default=6)
    parser.add_argument('--m-iters', type=int, default=300)
    parser.add_argument('--ik-iters', type=int, default=1000)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    if args.green_db is None:
        args.green_db = os.path.join(args.green_dir, 'green.duckdb')

    os.makedirs(args.output_dir, exist_ok=True)

    # ── Load Green data ─────────────────────────────────────────
    traj_path = os.path.join(args.green_dir, args.traj)
    print(f"Loading: {traj_path}")
    bindata, trials_index, traj3d = load_green_binary(traj_path)
    print(f"  {len(trials_index)} trials")

    import duckdb
    db = duckdb.connect(args.green_db, read_only=True)
    rats = [r[0] for r in db.sql("SELECT DISTINCT animal FROM trials ORDER BY animal").fetchall()]
    print(f"  Rats: {rats}")

    rat_trials = {}
    for rat in rats:
        tids = [r[0] for r in db.sql(f"SELECT id FROM trials WHERE animal='{rat}' ORDER BY id").fetchall()]
        rat_trials[rat] = tids
        print(f"    {rat}: {len(tids)} trials")

    # ── Load phase boundaries (arena → return = active behavior) ──
    print("\n  Loading phase boundaries...")
    phase_rows = db.sql(
        "SELECT id, arena_idx, return_idx FROM trials "
        "WHERE is_valid=1 AND arena_idx IS NOT NULL AND return_idx IS NOT NULL"
    ).fetchall()
    trial_phases = {r[0]: (int(r[1]), int(r[2])) for r in phase_rows}
    print(f"  {len(trial_phases)} trials with active phase boundaries")

    rng = np.random.default_rng(args.seed)

    # ── JAX info ────────────────────────────────────────────────
    import jax
    print(f"\nJAX devices: {jax.devices()}")

    # ── Fit 1: Average model (500 frames × 5 rats = 2500) ──────
    print(f"\n{'*'*70}")
    print(f"* AVERAGE MODEL: {args.frames_average} frames × {len(rats)} rats")
    print(f"{'*'*70}")

    all_frames = []
    for rat in rats:
        rat_frames = extract_frames(bindata, trials_index, traj3d,
                                     rat_trials[rat], args.frames_average, rng,
                                     trial_phases=trial_phases)
        print(f"  {rat}: {len(rat_frames)} frames")
        all_frames.extend(rat_frames)
    print(f"  Total: {len(all_frames)} frames")

    avg_output = os.path.join(args.output_dir, 'rodent_green_all.mjb')
    fit_on_frames(args.base_model, all_frames, avg_output, 'all_rats',
                  n_rounds=args.n_rounds, m_iters=args.m_iters, ik_iters=args.ik_iters)

    # ── Fit 2-6: Per-rat models (2500 frames each) ─────────────
    for rat in rats:
        print(f"\n{'*'*70}")
        print(f"* {rat.upper()}: {args.frames_per_rat} frames")
        print(f"{'*'*70}")

        rat_frames = extract_frames(bindata, trials_index, traj3d,
                                     rat_trials[rat], args.frames_per_rat, rng,
                                     trial_phases=trial_phases)
        print(f"  Extracted: {len(rat_frames)} frames")

        output = os.path.join(args.output_dir, f'rodent_green_{rat}.mjb')
        fit_on_frames(args.base_model, rat_frames, output, rat,
                      n_rounds=args.n_rounds, m_iters=args.m_iters, ik_iters=args.ik_iters)

    db.close()
    print(f"\n{'='*70}")
    print("ALL FITS COMPLETE")
    print(f"Output directory: {args.output_dir}")
    for f in sorted(os.listdir(args.output_dir)):
        if f.endswith('.mjb'):
            print(f"  {f}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
