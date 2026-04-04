"""CLI entry points for adjustabodies."""

import argparse
import os


def fit_cli():
    """CLI: Fit body model to data."""
    parser = argparse.ArgumentParser(description="Fit MuJoCo body model to keypoint data")
    parser.add_argument('--data-dir', required=True, help="RED project directory")
    parser.add_argument('--model-xml', required=True, help="Base MuJoCo XML model")
    parser.add_argument('--output', default=None, help="Output .mjb path")
    parser.add_argument('--max-frames', type=int, default=500)
    parser.add_argument('--ik-iters', type=int, default=1000)
    parser.add_argument('--n-rounds', type=int, default=6)
    parser.add_argument('--m-iters', type=int, default=300)
    args = parser.parse_args()

    if args.output is None:
        args.output = os.path.join(args.data_dir, "body_model_fitted.mjb")

    from .fit import fit_body_model
    fit_body_model(
        model_xml=args.model_xml,
        data_dir=args.data_dir,
        output_path=args.output,
        max_frames=args.max_frames,
        ik_iters=args.ik_iters,
        n_rounds=args.n_rounds,
        m_iters=args.m_iters,
    )


def ik_cli():
    """CLI: Batch IK export."""
    parser = argparse.ArgumentParser(description="Batch IK solve and export qpos")
    parser.add_argument('--data-dir', required=True)
    parser.add_argument('--model', required=True, help="Fitted .mjb or .xml model")
    parser.add_argument('--output', default=None, help="Output qpos CSV")
    parser.add_argument('--max-frames', type=int, default=0, help="0 = all frames")
    parser.add_argument('--ik-iters', type=int, default=1000)
    args = parser.parse_args()

    from .model import load_model, build_site_indices
    from .io import load_keypoints3d, find_keypoints3d
    from .arena import ArenaTransform
    from .ik_cpu import batch_ik_cpu
    from .species.rodent import RAT24_SITES
    import numpy as np

    m = load_model(args.model, add_free_joint=True)
    site_ids = build_site_indices(m, RAT24_SITES)

    session_path = os.path.join(args.data_dir, 'mujoco_session.json')
    arena_tf = ArenaTransform.from_session(session_path) if os.path.exists(session_path) else ArenaTransform()
    kp3d_csv = find_keypoints3d(args.data_dir)
    max_f = args.max_frames if args.max_frames > 0 else None
    frames = load_keypoints3d(kp3d_csv, max_frames=max_f, arena_tf=arena_tf)

    print(f"Solving IK on {len(frames)} frames...")
    all_qpos = batch_ik_cpu(m, frames, site_ids, max_iters=args.ik_iters)

    if args.output is None:
        args.output = os.path.join(args.data_dir, "qpos_export.csv")

    # Write CSV
    nq = m.nq
    with open(args.output, 'w') as f:
        f.write("frame")
        for j in range(nq):
            f.write(f",qpos_{j}")
        f.write("\n")
        for i in range(len(frames)):
            f.write(str(i))
            for j in range(nq):
                f.write(f",{all_qpos[i, j]:.8f}")
            f.write("\n")
    print(f"Saved: {args.output}")


def ik_mjx_cli():
    """CLI: GPU batch IK export using MJX."""
    parser = argparse.ArgumentParser(description="GPU batch IK solve via MJX")
    parser.add_argument('--data-dir', required=True)
    parser.add_argument('--model', required=True, help="Fitted .mjb or .xml model")
    parser.add_argument('--output', default=None, help="Output qpos CSV")
    parser.add_argument('--max-frames', type=int, default=0, help="0 = all frames")
    parser.add_argument('--ik-iters', type=int, default=1000)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--batch-size', type=int, default=512)
    args = parser.parse_args()

    from .model import load_model, build_site_indices
    from .io import load_keypoints3d, find_keypoints3d
    from .arena import ArenaTransform
    from .ik_mjx import batch_ik_mjx, IKConfig
    from .species.rodent import RAT24_SITES
    import numpy as np

    m = load_model(args.model, add_free_joint=True, fix_geoms_for_mjx=True)
    site_ids = build_site_indices(m, RAT24_SITES)

    session_path = os.path.join(args.data_dir, 'mujoco_session.json')
    arena_tf = ArenaTransform.from_session(session_path) if os.path.exists(session_path) else ArenaTransform()
    kp3d_csv = find_keypoints3d(args.data_dir)
    max_f = args.max_frames if args.max_frames > 0 else None
    frames = load_keypoints3d(kp3d_csv, max_frames=max_f, arena_tf=arena_tf)

    config = IKConfig(max_iters=args.ik_iters, lr=args.lr, batch_size=args.batch_size)
    all_qpos, residuals = batch_ik_mjx(m, frames, site_ids, config=config)

    if args.output is None:
        args.output = os.path.join(args.data_dir, "qpos_export_mjx.csv")

    nq = m.nq
    with open(args.output, 'w') as f:
        f.write("frame")
        for j in range(nq):
            f.write(f",qpos_{j}")
        f.write(",residual_mm\n")
        for i in range(len(frames)):
            f.write(str(i))
            for j in range(nq):
                f.write(f",{all_qpos[i, j]:.8f}")
            f.write(f",{residuals[i]:.4f}\n")
    print(f"Saved: {args.output}")
