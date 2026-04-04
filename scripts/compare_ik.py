#!/usr/bin/env python3
"""Compare CPU and MJX IK solvers on the same dataset.

Loads keypoint data from a RED project, runs both solvers, and reports
residual statistics and timing. Optionally compares against an existing
qpos_export.csv reference.

Usage:
    python scripts/compare_ik.py \
        --data-dir /Volumes/johnsonlab/virtual_rodent/tiny_project \
        --model /Volumes/johnsonlab/virtual_rodent/tiny_project/rodent_mjx_fitted.mjb \
        --max-frames 100

    python scripts/compare_ik.py \
        --data-dir /Volumes/johnsonlab/virtual_rodent/tiny_project \
        --model /Volumes/johnsonlab/virtual_rodent/tiny_project/rodent_mjx_fitted.mjb \
        --reference-csv /Volumes/johnsonlab/virtual_rodent/tiny_project/qpos_export.csv
"""

import argparse
import os
import sys
import time
import numpy as np


def compute_residuals_cpu(m, frames, site_ids, all_qpos):
    """Compute per-frame RMSE residuals for given qpos solutions."""
    import mujoco
    d = mujoco.MjData(m)
    residuals = np.zeros(len(frames))

    for i, (kp_mj, valid) in enumerate(frames):
        d.qpos[:] = all_qpos[i]
        mujoco.mj_fwdPosition(m, d)

        err_sq_sum = 0.0
        n_valid = 0
        for k in range(24):
            if valid[k] > 0.5 and site_ids[k] >= 0:
                sid = site_ids[k]
                diff = d.site_xpos[sid] - kp_mj[k]
                err_sq_sum += np.sum(diff ** 2)
                n_valid += 1
        if n_valid > 0:
            residuals[i] = np.sqrt(err_sq_sum / n_valid) * 1000.0  # m → mm
    return residuals


def print_residual_stats(name, residuals_mm):
    """Print residual statistics."""
    valid = residuals_mm[residuals_mm > 0]
    if len(valid) == 0:
        print(f"  {name}: no valid frames")
        return
    print(f"  {name}: mean={valid.mean():.2f}mm  median={np.median(valid):.2f}mm  "
          f"p95={np.percentile(valid, 95):.2f}mm  max={valid.max():.2f}mm  "
          f"({len(valid)} frames)")


def main():
    parser = argparse.ArgumentParser(description="Compare CPU vs MJX IK")
    parser.add_argument('--data-dir', required=True, help="RED project directory")
    parser.add_argument('--model', required=True, help="Fitted .mjb or .xml model")
    parser.add_argument('--max-frames', type=int, default=0, help="0 = all")
    parser.add_argument('--ik-iters', type=int, default=1000)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--batch-size', type=int, default=512)
    parser.add_argument('--reference-csv', default=None, help="Existing qpos_export.csv")
    parser.add_argument('--cpu-only', action='store_true', help="Skip MJX (CPU test only)")
    parser.add_argument('--mjx-only', action='store_true', help="Skip CPU (MJX test only)")
    args = parser.parse_args()

    # ── Load data ──────────────────────────────────────────────────────
    from adjustabodies.model import load_model, build_site_indices
    from adjustabodies.io import load_keypoints3d, find_keypoints3d
    from adjustabodies.arena import ArenaTransform
    from adjustabodies.species.rodent import RAT24_SITES

    print(f"Loading model: {args.model}")
    m = load_model(args.model, add_free_joint=True)
    site_ids = build_site_indices(m, RAT24_SITES)
    mapped = sum(1 for s in site_ids if s >= 0)
    print(f"  nq={m.nq} nv={m.nv} sites={mapped}/24 mapped")

    session_path = os.path.join(args.data_dir, 'mujoco_session.json')
    if os.path.exists(session_path):
        arena_tf = ArenaTransform.from_session(session_path)
        print(f"  Arena transform: scale={arena_tf.scale}")
    else:
        arena_tf = ArenaTransform()
        print("  No arena transform (identity + mm→m)")

    kp3d_csv = find_keypoints3d(args.data_dir)
    if kp3d_csv is None:
        print(f"ERROR: no keypoints3d.csv found in {args.data_dir}")
        sys.exit(1)
    print(f"Loading keypoints: {kp3d_csv}")

    max_f = args.max_frames if args.max_frames > 0 else None
    frames = load_keypoints3d(kp3d_csv, max_frames=max_f, arena_tf=arena_tf)
    N = len(frames)
    print(f"  {N} frames loaded")

    # Count valid keypoints
    n_valid_per_frame = np.array([v.sum() for _, v in frames])
    print(f"  Valid keypoints: mean={n_valid_per_frame.mean():.1f} "
          f"min={n_valid_per_frame.min():.0f} max={n_valid_per_frame.max():.0f}")

    # ── CPU IK ─────────────────────────────────────────────────────────
    cpu_qpos = None
    cpu_residuals = None
    cpu_time = 0.0

    if not args.mjx_only:
        from adjustabodies.ik_cpu import batch_ik_cpu

        print(f"\n{'='*60}")
        print(f"CPU IK (max_iters={args.ik_iters}, lr={args.lr})")
        print(f"{'='*60}")

        t0 = time.time()
        cpu_qpos = batch_ik_cpu(m, frames, site_ids,
                                max_iters=args.ik_iters, lr=args.lr)
        cpu_time = time.time() - t0

        cpu_residuals = compute_residuals_cpu(m, frames, site_ids, cpu_qpos)
        print(f"  Time: {cpu_time:.1f}s ({N/max(cpu_time, 0.001):.1f} fps)")
        print_residual_stats("CPU", cpu_residuals)

    # ── MJX IK ─────────────────────────────────────────────────────────
    mjx_qpos = None
    mjx_residuals = None
    mjx_time = 0.0

    if not args.cpu_only:
        from adjustabodies.ik_mjx import batch_ik_mjx, IKConfig
        import mujoco

        print(f"\n{'='*60}")
        print(f"MJX IK (max_iters={args.ik_iters}, lr={args.lr}, "
              f"batch_size={args.batch_size})")
        print(f"{'='*60}")

        # Load a separate model copy for MJX (fix_geoms modifies in place)
        m_mjx = load_model(args.model, add_free_joint=True,
                           fix_geoms_for_mjx=True)
        site_ids_mjx = build_site_indices(m_mjx, RAT24_SITES)

        # Auto-detect: use scan mode on GPU, step mode on CPU
        import jax
        has_gpu = any(d.platform == 'gpu' for d in jax.devices())
        mjx_config = IKConfig(
            max_iters=args.ik_iters,
            lr=args.lr,
            batch_size=args.batch_size,
            use_scan=has_gpu,
        )

        t0 = time.time()
        mjx_qpos, mjx_residuals = batch_ik_mjx(
            m_mjx, frames, site_ids_mjx, config=mjx_config)
        mjx_time = time.time() - t0

        print(f"  Total time: {mjx_time:.1f}s")
        print_residual_stats("MJX", mjx_residuals)

    # ── Reference CSV ──────────────────────────────────────────────────
    ref_residuals = None
    if args.reference_csv and os.path.exists(args.reference_csv):
        from adjustabodies.io import load_qpos_export

        print(f"\n{'='*60}")
        print(f"Reference: {args.reference_csv}")
        print(f"{'='*60}")

        ref_dict, ref_nq = load_qpos_export(args.reference_csv)
        # Map reference frames to our frame indices
        ref_qpos = np.zeros((N, m.nq), dtype=np.float64)
        ref_matched = 0
        for i in range(N):
            if i in ref_dict:
                q = ref_dict[i]
                ref_qpos[i, :len(q)] = q
                ref_matched += 1
        print(f"  Matched {ref_matched}/{N} frames (ref nq={ref_nq})")

        if ref_matched > 0:
            ref_residuals = compute_residuals_cpu(m, frames, site_ids, ref_qpos)
            print_residual_stats("Reference", ref_residuals)

    # ── Comparison ─────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("COMPARISON")
    print(f"{'='*60}")

    results = {}
    if cpu_residuals is not None:
        results['CPU'] = cpu_residuals
    if mjx_residuals is not None:
        results['MJX'] = mjx_residuals
    if ref_residuals is not None:
        results['Reference'] = ref_residuals

    # Pairwise comparisons
    names = list(results.keys())
    for i in range(len(names)):
        for j in range(i+1, len(names)):
            a_name, b_name = names[i], names[j]
            a, b = results[a_name], results[b_name]
            # Only compare frames where both have valid residuals
            mask = (a > 0) & (b > 0)
            if mask.sum() == 0:
                continue
            diff = a[mask] - b[mask]
            abs_diff = np.abs(diff)
            print(f"\n  {a_name} vs {b_name} ({mask.sum()} frames):")
            print(f"    {a_name} mean={a[mask].mean():.2f}mm  "
                  f"{b_name} mean={b[mask].mean():.2f}mm")
            print(f"    Diff (A-B): mean={diff.mean():.3f}mm  "
                  f"median={np.median(diff):.3f}mm  max_abs={abs_diff.max():.3f}mm")
            corr = np.corrcoef(a[mask], b[mask])[0, 1]
            print(f"    Correlation: {corr:.4f}")

    # Qpos comparison (hinge joints only, skip free joint)
    if cpu_qpos is not None and mjx_qpos is not None:
        # Compare hinge joint angles (qpos[7:] for models with free joint)
        hinge_start = 7 if any(m.jnt_type[j] == 0 for j in range(m.njnt)) else 0
        cpu_hinges = cpu_qpos[:, hinge_start:]
        mjx_hinges = mjx_qpos[:, hinge_start:]
        hinge_diff = np.abs(cpu_hinges - mjx_hinges)
        print(f"\n  Hinge joint angle difference (CPU vs MJX):")
        print(f"    Mean: {hinge_diff.mean():.4f} rad ({np.degrees(hinge_diff.mean()):.2f} deg)")
        print(f"    Max:  {hinge_diff.max():.4f} rad ({np.degrees(hinge_diff.max()):.2f} deg)")

    # Timing comparison
    if cpu_time > 0 and mjx_time > 0:
        print(f"\n  Timing:")
        print(f"    CPU: {cpu_time:.1f}s ({N/max(cpu_time, 0.001):.1f} fps)")
        print(f"    MJX: {mjx_time:.1f}s ({N/max(mjx_time, 0.001):.1f} fps)")
        print(f"    Speedup: {cpu_time/max(mjx_time, 0.001):.1f}x")

    print()


if __name__ == "__main__":
    main()
