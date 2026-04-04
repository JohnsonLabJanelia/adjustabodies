"""Complete body model fitting pipeline: resize + STAC → fitted MJB.

Usage:
    from adjustabodies.fit import fit_body_model
    result = fit_body_model(model_xml, data_dir, output_mjb)
"""

import os
import json
import time
import numpy as np
import mujoco

from .model import load_model, build_segment_indices, build_site_indices, save_originals
from .io import load_keypoints3d, find_keypoints3d, save_fitted_model
from .arena import ArenaTransform
from .species.rodent import (SEGMENT_DEFS, RAT24_SITES, SEGMENT_LENGTH_INIT,
                              LR_SITE_PAIRS, MIDLINE_SITES)


def fit_body_model(model_xml: str, data_dir: str, output_path: str,
                    max_frames: int = 500,
                    ik_iters: int = 1000,
                    n_rounds: int = 6,
                    m_iters: int = 300,
                    verbose: bool = True) -> dict:
    """Run the complete 2-phase body model fitting pipeline.

    Phase 1: Body segment scaling (no site offsets)
    Phase 2: STAC site offsets with L/R symmetry

    Args:
        model_xml: Path to base MuJoCo XML model
        data_dir: RED project directory with labeled_data/ and mujoco_session.json
        output_path: Where to save the fitted .mjb file
        max_frames: Number of frames to use for fitting
        ik_iters: IK iterations per frame in Q-phase
        n_rounds: Alternating Q/M rounds per phase
        m_iters: Adam steps per M-phase round

    Returns:
        dict with fitting results (scales, offsets, residuals)
    """
    # Lazy import MJX (only needed on GPU)
    import jax
    import jax.numpy as jnp
    from mujoco import mjx
    from .resize import run_resize_phase, build_mjx_scale_fn
    from .stac import run_stac_phase

    if verbose:
        print(f"JAX devices: {jax.devices()}")

    # Load model
    m = load_model(model_xml, add_free_joint=True, fix_geoms_for_mjx=True)
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    if verbose:
        print(f"Model: nq={m.nq}, nbody={m.nbody}, nsite={m.nsite}")

    segments = build_segment_indices(m, SEGMENT_DEFS)
    site_ids = build_site_indices(m, RAT24_SITES)
    orig = save_originals(m)
    n_seg = len(segments)

    # Load data
    session_path = os.path.join(data_dir, 'mujoco_session.json')
    arena_tf = ArenaTransform.from_session(session_path) if os.path.exists(session_path) else ArenaTransform()
    kp3d_csv = find_keypoints3d(data_dir)
    assert kp3d_csv, f"No keypoints3d.csv found in {data_dir}"
    frames = load_keypoints3d(kp3d_csv, max_frames=max_frames, arena_tf=arena_tf)
    if verbose:
        print(f"Frames: {len(frames)}")

    # Build MJX model and scale function
    mx_base = mjx.put_model(m)
    apply_scales_fn = build_mjx_scale_fn(m, segments, orig)

    # Segment length initialization
    init_rel_scales = np.array(
        [SEGMENT_LENGTH_INIT.get(name, 1.0) for name, _ in segments],
        dtype=np.float32)

    # ════════════════════════════════════════════════════════════════
    # PHASE 1: Body segment scaling
    # ════════════════════════════════════════════════════════════════
    if verbose:
        print(f"\n{'='*60}")
        print("PHASE 1: Body segment scaling (no site offsets)")
        print(f"{'='*60}")

    params, pre_res, post_res_1 = run_resize_phase(
        m, mx_base, segments, site_ids, orig, frames, apply_scales_fn,
        init_global=1.0, init_rel_scales=init_rel_scales,
        n_rounds=n_rounds, m_iters=m_iters, ik_iters=ik_iters,
        lr_scale=0.003, reg_scale=0.001, verbose=verbose)

    if verbose:
        gs = float(params['global_scale'])
        rs = np.array(params['rel_scales'])
        print(f"\nPhase 1 scales:")
        for g, (name, _) in enumerate(segments):
            print(f"  {name:<12s} {gs*rs[g]:.3f} (rel={rs[g]:.3f})")

    # ════════════════════════════════════════════════════════════════
    # PHASE 2: STAC site offsets with L/R symmetry
    # ════════════════════════════════════════════════════════════════
    if verbose:
        print(f"\n{'='*60}")
        print("PHASE 2: Site offset calibration (STAC) with L/R symmetry")
        print(f"{'='*60}")

    # Need to rebuild MJX model with Phase 1 scales applied
    from .model import apply_segment_scales
    gs = float(params['global_scale'])
    rs = np.array(params['rel_scales'])
    scales = {name: gs * rs[g] for g, (name, _) in enumerate(segments)}
    apply_segment_scales(m, segments, scales, orig)
    # Update originals to include scales (Phase 2 starts from scaled model)
    orig_scaled = save_originals(m)
    mx_base_scaled = mjx.put_model(m)
    apply_scales_fn_2 = build_mjx_scale_fn(m, segments, orig_scaled)

    # Reset rel_scales to 1.0 for Phase 2 (scales are baked into originals)
    params['rel_scales'] = jnp.ones(n_seg)
    params['global_scale'] = jnp.array(1.0)

    params, pre_res_2, post_res_2 = run_stac_phase(
        m, mx_base_scaled, segments, site_ids, orig_scaled, frames,
        apply_scales_fn_2, params,
        sym_config=(LR_SITE_PAIRS, MIDLINE_SITES),
        n_rounds=n_rounds, m_iters=m_iters, ik_iters=ik_iters,
        lr_scale=0.0001, lr_offset=0.001,
        reg_scale=10.0, reg_offset=0.01, verbose=verbose)

    # ════════════════════════════════════════════════════════════════
    # Save results
    # ════════════════════════════════════════════════════════════════
    if verbose:
        print(f"\n{'='*60}")
        print("SAVING RESULTS")
        print(f"{'='*60}")

    # Apply final state to CPU model
    gs_final = float(params['global_scale'])
    rs_final = np.array(params['rel_scales'])
    offs_final = np.array(params['site_offsets'])
    final_scales = {name: gs_final * rs_final[g] for g, (name, _) in enumerate(segments)}
    apply_segment_scales(m, segments, final_scales, orig_scaled)
    m.site_pos[:] += offs_final
    mujoco.mj_setConst(m, mujoco.MjData(m))

    # Compute final absolute scales (Phase 1 × Phase 2)
    abs_scales = {}
    for g, (name, _) in enumerate(segments):
        abs_scales[name] = scales[name] * final_scales[name]

    # Metadata
    disps = np.linalg.norm(offs_final, axis=1) * 1000
    metadata = {
        'adjustabodies_version': '0.1.0',
        'base_model': model_xml,
        'data_source': data_dir,
        'n_frames': len(frames),
        'phase1_residual_mm': post_res_1,
        'phase2_residual_mm': post_res_2,
        'segment_scales': abs_scales,
        'top_site_offsets': {m.site(i).name: disps[i]
                             for i in np.argsort(disps)[::-1][:10] if disps[i] > 0.1},
    }

    save_fitted_model(m, output_path, metadata)

    if verbose:
        print(f"\nFinal segment scales:")
        for name, s in abs_scales.items():
            print(f"  {name:<12s} {s:.3f}")
        print(f"\nTop site offsets:")
        for name, d in metadata['top_site_offsets'].items():
            print(f"  {name:<30s} {d:.1f} mm")
        print(f"\nResidual: {pre_res:.2f} → {post_res_1:.2f} → {post_res_2:.2f} mm")
        print(f"Saved: {output_path}")

    return metadata
