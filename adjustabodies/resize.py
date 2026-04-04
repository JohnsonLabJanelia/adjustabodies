"""Body segment scaling via MJX (GPU) — Phase 1 of the fitting pipeline.

Alternating optimization:
  Q-phase: IK solve per frame (CPU, MuJoCo C API)
  M-phase: Adam on segment scale factors (GPU, MJX differentiable FK)
"""

import numpy as np
import time
from typing import List, Tuple, Dict, Optional

try:
    import jax
    import jax.numpy as jnp
    import optax
    from mujoco import mjx
    HAS_MJX = True
except ImportError:
    HAS_MJX = False

from .model import load_model, build_segment_indices, build_site_indices, save_originals
from .ik_cpu import batch_ik_cpu


def build_mjx_scale_fn(m, segments, orig):
    """Build JAX-compatible function that applies segment scales to MJX model.

    Returns: apply_scales(mx, global_scale, rel_scales, site_offsets) -> mx
    """
    n_seg = len(segments)
    body_to_seg = np.full(m.nbody, -1, dtype=np.int32)
    for g, (_, bids) in enumerate(segments):
        for bid in bids:
            body_to_seg[bid] = g

    bts_j = jnp.array(body_to_seg)
    gb_j = jnp.array(m.geom_bodyid)
    sb_j = jnp.array(m.site_bodyid)
    jb_j = jnp.array(m.jnt_bodyid)

    orig_j = {k: jnp.array(v) for k, v in orig.items()}

    def apply_scales(mx, global_scale, rel_scales, site_offsets):
        ss = global_scale * rel_scales
        bs = jnp.where(bts_j >= 0, ss[jnp.clip(bts_j, 0, n_seg - 1)], 1.0)
        return mx.replace(
            body_pos=orig_j['body_pos'] * bs[:, None],
            body_ipos=orig_j['body_ipos'] * bs[:, None],
            geom_pos=orig_j['geom_pos'] * bs[gb_j][:, None],
            geom_size=orig_j['geom_size'] * bs[gb_j][:, None],
            site_pos=orig_j['site_pos'] * bs[sb_j][:, None] + site_offsets,
            jnt_pos=orig_j['jnt_pos'] * bs[jb_j][:, None],
        )

    return apply_scales


def run_resize_phase(m, mx_base, segments, site_ids, orig,
                      frames, apply_scales_fn,
                      init_global=1.0, init_rel_scales=None,
                      n_rounds=6, m_iters=300, ik_iters=1000,
                      lr_scale=0.003, reg_scale=0.001,
                      verbose=True):
    """Phase 1: Optimize segment scales (no site offsets).

    Returns: (global_scale, rel_scales, pre_residual, post_residual)
    """
    n_seg = len(segments)
    nq = m.nq
    N = len(frames)
    nsite = m.nsite

    si_j = jnp.array(site_ids)

    if init_rel_scales is None:
        init_rel_scales = np.ones(n_seg, dtype=np.float32)

    params = {
        'global_scale': jnp.array(float(init_global)),
        'rel_scales': jnp.array(init_rel_scales.astype(np.float32)),
        'site_offsets': jnp.zeros((nsite, 3)),
    }

    optimizer = optax.adam(lr_scale)
    opt_state = optimizer.init(params)

    kp3d_all = jnp.array(np.stack([f[0] for f in frames]))
    valid_all = jnp.array(np.stack([f[1] for f in frames]))

    def fk_sites(mx, qpos):
        dx = mjx.make_data(mx).replace(qpos=qpos)
        dx = mjx.kinematics(mx, dx)
        dx = mjx.com_pos(mx, dx)
        return dx.site_xpos

    def m_loss(params, qpos_j, kp3d, valid):
        mx_s = apply_scales_fn(mx_base, params['global_scale'],
                                params['rel_scales'], jnp.zeros((nsite, 3)))
        all_sites = jax.vmap(lambda q: fk_sites(mx_s, q))(qpos_j)
        kp_sites = all_sites[:, si_j, :]
        res = (kp_sites - kp3d) * valid[:, :, None]
        ik_loss = jnp.mean(jnp.sum(res ** 2, axis=-1))
        r_s = reg_scale * jnp.sum((params['rel_scales'] - 1.0) ** 2)
        return ik_loss + r_s, {'ik': ik_loss}

    @jax.jit
    def m_step(params, opt_state, qpos_j, kp3d, valid):
        (loss, metrics), grads = jax.value_and_grad(m_loss, has_aux=True)(
            params, qpos_j, kp3d, valid)
        updates, new_opt = optimizer.update(grads, opt_state, params)
        new_params = optax.apply_updates(params, updates)
        new_params['global_scale'] = jnp.clip(new_params['global_scale'], 0.7, 1.5)
        new_params['rel_scales'] = jnp.clip(new_params['rel_scales'], 0.5, 1.5)
        return new_params, new_opt, loss, metrics

    # Compile
    if verbose:
        print("Compiling resize phase...", end=" ", flush=True)
    t0 = time.time()
    dummy = jnp.zeros((N, nq))
    m_step(params, opt_state, dummy, kp3d_all, valid_all)
    if verbose:
        print(f"{time.time() - t0:.1f}s")

    # Helper to apply scales to CPU model
    def apply_to_cpu():
        from .model import apply_segment_scales
        gs = float(params['global_scale'])
        rs = np.array(params['rel_scales'])
        scales = {name: gs * rs[g] for g, (name, _) in enumerate(segments)}
        apply_segment_scales(m, segments, scales, orig)

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
        gs = float(params['global_scale'])
        rs = np.array(params['rel_scales'])
        changed = [(seg_names[g], rs[g]) for g in range(n_seg) if abs(rs[g] - 1.0) > 0.01]
        ch_str = ", ".join(f"{n}={r:.3f}" for n, r in changed[:6]) if changed else "(~1.0)"

        if verbose:
            print(f"  Round {rnd+1}/{n_rounds}: IK={ik_mm:.2f}mm Q={dt_q:.0f}s M={dt_m:.1f}s "
                  f"global={gs:.3f} {ch_str}")

    return params, pre_residual, ik_mm
