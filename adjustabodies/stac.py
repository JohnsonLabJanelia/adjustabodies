"""STAC site offset calibration via MJX (GPU) — Phase 2 of the fitting pipeline.

Alternating optimization with frozen body scales:
  Q-phase: IK solve per frame (CPU)
  M-phase: Adam on site offsets with L/R symmetry (GPU, MJX FK)
"""

import numpy as np
import time
from typing import List, Optional

try:
    import jax
    import jax.numpy as jnp
    import optax
    from mujoco import mjx
    HAS_MJX = True
except ImportError:
    HAS_MJX = False

from .ik_cpu import batch_ik_cpu
from .symmetry import enforce_symmetry_jax, build_symmetry_indices_jax


def run_stac_phase(m, mx_base, segments, site_ids, orig,
                    frames, apply_scales_fn, params,
                    sym_config=None,
                    n_rounds=6, m_iters=300, ik_iters=1000,
                    lr_scale=0.0001, lr_offset=0.001,
                    reg_scale=10.0, reg_offset=0.01,
                    verbose=True):
    """Phase 2: Optimize site offsets (STAC) on the resized body.

    Scales are mostly frozen (high reg + tiny LR). Site offsets are the
    primary optimization target, with L/R symmetry enforcement.

    Args:
        params: dict with 'global_scale', 'rel_scales', 'site_offsets' from Phase 1
        sym_config: tuple (site_pairs, midline_sites) for symmetry enforcement

    Returns: (params, pre_residual, post_residual)
    """
    n_seg = len(segments)
    nq = m.nq
    N = len(frames)
    nsite = m.nsite

    si_j = jnp.array(site_ids)
    kp3d_all = jnp.array(np.stack([f[0] for f in frames]))
    valid_all = jnp.array(np.stack([f[1] for f in frames]))

    # Build symmetry indices
    sym_mid, sym_L, sym_R = None, None, None
    if sym_config:
        site_pairs, midline_sites = sym_config
        sym_mid, sym_L, sym_R = build_symmetry_indices_jax(m, site_pairs, midline_sites)
        if verbose:
            print(f"  Symmetry: {len(sym_mid)} midline, {len(sym_L)} L/R pairs")

    # Scales are frozen in Phase 2 — only site offsets are optimized
    optimizer = optax.multi_transform(
        {'scales': optax.set_to_zero(), 'offsets': optax.adam(lr_offset)},
        param_labels={'global_scale': 'scales', 'rel_scales': 'scales',
                      'site_offsets': 'offsets'})
    opt_state = optimizer.init(params)

    def fk_sites(mx, qpos):
        dx = mjx.make_data(mx).replace(qpos=qpos)
        dx = mjx.kinematics(mx, dx)
        dx = mjx.com_pos(mx, dx)
        return dx.site_xpos

    def m_loss(params, qpos_j, kp3d, valid):
        mx_s = apply_scales_fn(mx_base, params['global_scale'],
                                params['rel_scales'], params['site_offsets'])
        all_sites = jax.vmap(lambda q: fk_sites(mx_s, q))(qpos_j)
        kp_sites = all_sites[:, si_j, :]
        res = (kp_sites - kp3d) * valid[:, :, None]
        ik_loss = jnp.mean(jnp.sum(res ** 2, axis=-1))
        r_o = reg_offset * jnp.sum(params['site_offsets'] ** 2)
        return ik_loss + r_o, {'ik': ik_loss}

    use_symmetry = sym_mid is not None

    @jax.jit
    def m_step(params, opt_state, qpos_j, kp3d, valid):
        (loss, metrics), grads = jax.value_and_grad(m_loss, has_aux=True)(
            params, qpos_j, kp3d, valid)
        updates, new_opt = optimizer.update(grads, opt_state, params)
        new_params = optax.apply_updates(params, updates)
        if use_symmetry:
            new_params['site_offsets'] = enforce_symmetry_jax(
                new_params['site_offsets'], sym_mid, sym_L, sym_R)
        return new_params, new_opt, loss, metrics

    # Compile
    if verbose:
        print("Compiling STAC phase...", end=" ", flush=True)
    t0 = time.time()
    dummy = jnp.zeros((N, nq))
    m_step(params, opt_state, dummy, kp3d_all, valid_all)
    if verbose:
        print(f"{time.time() - t0:.1f}s")

    def apply_to_cpu():
        from .model import apply_segment_scales
        gs = float(params['global_scale'])
        rs = np.array(params['rel_scales'])
        offs = np.array(params['site_offsets'])
        scales = {name: gs * rs[g] for g, (name, _) in enumerate(segments)}
        apply_segment_scales(m, segments, scales, orig)
        m.site_pos[:] += offs
        import mujoco
        mujoco.mj_setConst(m, mujoco.MjData(m))

    pre_residual = None
    seg_names = [s[0] for s in segments]

    for rnd in range(n_rounds):
        t_q = time.time()
        apply_to_cpu()
        all_qpos = batch_ik_cpu(m, frames, site_ids, max_iters=ik_iters)
        all_qpos_j = jnp.array(all_qpos)
        dt_q = time.time() - t_q

        t_m = time.time()
        opt_state = optimizer.init(params)
        for _ in range(m_iters):
            params, opt_state, loss, metrics = m_step(
                params, opt_state, all_qpos_j, kp3d_all, valid_all)
        dt_m = time.time() - t_m

        ik_mm = float(jnp.sqrt(metrics['ik'] / 24) * 1000)
        if rnd == 0:
            pre_residual = ik_mm

        if verbose:
            gs = float(params['global_scale'])
            rs = np.array(params['rel_scales'])
            changed = [(seg_names[g], rs[g]) for g in range(n_seg) if abs(rs[g] - 1.0) > 0.01]
            ch_str = ", ".join(f"{n}={r:.3f}" for n, r in changed[:6]) if changed else "(~1.0)"
            print(f"  Round {rnd+1}/{n_rounds}: IK={ik_mm:.2f}mm Q={dt_q:.0f}s M={dt_m:.1f}s "
                  f"global={gs:.3f} {ch_str}")

    return params, pre_residual, ik_mm
