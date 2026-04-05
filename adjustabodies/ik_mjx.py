"""GPU-accelerated IK solver using MuJoCo MJX (JAX).

Velocity-space gradient descent with proper quaternion integration.
The key idea: compute gradients w.r.t. a velocity perturbation (nv-dim),
then apply the step using integrate_pos_jax which correctly handles
quaternion updates via the exponential map. This avoids the broken
naive gradient descent on quaternion components that caused 27mm
residuals in the original implementation.

Requires: pip install "adjustabodies[mjx]"
"""

from typing import List, Tuple, Optional, NamedTuple
import numpy as np
from . import enable_jax_cache


class IKConfig(NamedTuple):
    """IK solver hyperparameters."""
    max_iters: int = 1000
    lr: float = 0.001
    beta: float = 0.99
    reg: float = 1e-4
    batch_size: int = 512
    use_scan: bool = True  # False = Python loop (fast compile, slower run)


class JointInfo(NamedTuple):
    """Precomputed joint topology arrays (all JAX arrays)."""
    jnt_type: object       # [njnt] int
    jnt_qposadr: object    # [njnt] int
    jnt_dofadr: object     # [njnt] int
    jnt_limited: object    # [njnt] bool
    jnt_range: object      # [njnt, 2] float
    hinge_qa: object       # [n_hinge] int — qpos indices for hinge joints
    hinge_va: object       # [n_hinge] int — vel indices for hinge joints
    free_joint_qa: int     # qposadr of first free joint (-1 if none)
    free_joint_va: int     # dofadr of first free joint (-1 if none)
    clamp_lo: object       # [nq] float — joint lower limits (-1e30 for unclamped)
    clamp_hi: object       # [nq] float — joint upper limits (+1e30 for unclamped)
    qpos0: object          # [nq] float — default pose
    nq: int
    nv: int


# ── Quaternion helpers (MuJoCo [w, x, y, z] convention) ──────────────

def _import_jax():
    import jax
    import jax.numpy as jnp
    return jax, jnp


def quat_multiply(q1, q2):
    """Hamilton product of two quaternions in [w, x, y, z] format."""
    _, jnp = _import_jax()
    w1, x1, y1, z1 = q1[0], q1[1], q1[2], q1[3]
    w2, x2, y2, z2 = q2[0], q2[1], q2[2], q2[3]
    return jnp.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
    ])


def axangle_to_quat(omega):
    """Convert axis-angle vector (3D) to unit quaternion [w, x, y, z].

    Numerically stable near zero rotation. Uses safe norm (eps) to avoid
    NaN gradients when omega is exactly zero.
    """
    _, jnp = _import_jax()
    # Safe norm: jnp.linalg.norm has undefined gradient at zero
    angle_sq = jnp.sum(omega ** 2)
    angle = jnp.sqrt(angle_sq + 1e-20)  # eps prevents 0/0 in gradient
    half_angle = angle / 2.0
    # For small angles: sin(a/2)/a ≈ 0.5 - a^2/48
    s = jnp.where(angle_sq > 1e-20,
                  jnp.sin(half_angle) / angle,
                  0.5 - angle_sq / 48.0)
    w = jnp.where(angle_sq > 1e-20,
                  jnp.cos(half_angle),
                  1.0 - angle_sq / 8.0)
    return jnp.concatenate([w[None], s * omega])


# ── Joint topology extraction ────────────────────────────────────────

def build_joint_info(m) -> JointInfo:
    """Extract joint topology from MjModel into JAX arrays.

    Args:
        m: mujoco.MjModel (CPU model)

    Returns:
        JointInfo with all arrays as jax.numpy arrays
    """
    _, jnp = _import_jax()

    nq, nv, njnt = m.nq, m.nv, m.njnt

    # Find hinge joints (type 3) — build vectorized index arrays
    hinge_qa_list = []
    hinge_va_list = []
    free_joint_qa = -1
    free_joint_va = -1

    for j in range(njnt):
        if m.jnt_type[j] == 0:  # FREE
            if free_joint_qa < 0:
                free_joint_qa = int(m.jnt_qposadr[j])
                free_joint_va = int(m.jnt_dofadr[j])
        elif m.jnt_type[j] == 3:  # HINGE
            hinge_qa_list.append(int(m.jnt_qposadr[j]))
            hinge_va_list.append(int(m.jnt_dofadr[j]))
        elif m.jnt_type[j] == 2:  # SLIDE
            hinge_qa_list.append(int(m.jnt_qposadr[j]))
            hinge_va_list.append(int(m.jnt_dofadr[j]))

    # Build clamp arrays for all qpos indices
    clamp_lo = np.full(nq, -1e30)
    clamp_hi = np.full(nq, 1e30)
    for j in range(njnt):
        if m.jnt_limited[j] and m.jnt_type[j] in (2, 3):  # SLIDE or HINGE
            qa = int(m.jnt_qposadr[j])
            clamp_lo[qa] = m.jnt_range[j, 0]
            clamp_hi[qa] = m.jnt_range[j, 1]

    return JointInfo(
        jnt_type=jnp.array(m.jnt_type),
        jnt_qposadr=jnp.array(m.jnt_qposadr),
        jnt_dofadr=jnp.array(m.jnt_dofadr),
        jnt_limited=jnp.array(m.jnt_limited),
        jnt_range=jnp.array(m.jnt_range),
        hinge_qa=jnp.array(hinge_qa_list, dtype=jnp.int32),
        hinge_va=jnp.array(hinge_va_list, dtype=jnp.int32),
        free_joint_qa=free_joint_qa,
        free_joint_va=free_joint_va,
        clamp_lo=jnp.array(clamp_lo),
        clamp_hi=jnp.array(clamp_hi),
        qpos0=jnp.array(m.qpos0),
        nq=nq,
        nv=nv,
    )


# ── Velocity-space integration (JAX equivalent of mj_integratePos) ───

def integrate_pos_jax(qpos, vel, dt, ji: JointInfo):
    """Apply a velocity step to qpos with proper quaternion integration.

    This is the JAX equivalent of mujoco.mj_integratePos. For hinge/slide
    joints it's a simple Euler step. For the free joint quaternion, it uses
    the exponential map: q_new = q_old * axangle_to_quat(omega * dt).

    Args:
        qpos: [nq] current joint positions
        vel: [nv] velocity step
        dt: time step (typically 1.0)
        ji: JointInfo with precomputed topology

    Returns:
        [nq] updated joint positions
    """
    _, jnp = _import_jax()

    # Hinge/slide joints: direct Euler step (vectorized)
    new_qpos = qpos.at[ji.hinge_qa].add(vel[ji.hinge_va] * dt)

    # Free joint (if present)
    if ji.free_joint_qa >= 0:
        qa = ji.free_joint_qa
        va = ji.free_joint_va

        # Translation: pos += vel * dt
        new_qpos = new_qpos.at[qa:qa+3].add(vel[va:va+3] * dt)

        # Rotation: q_new = q_old * axangle_to_quat(omega * dt)
        omega = vel[va+3:va+6] * dt
        dq = axangle_to_quat(omega)
        q_old = new_qpos[qa+3:qa+7]
        q_new = quat_multiply(q_old, dq)
        # Normalize for numerical safety (safe norm for gradient)
        q_new = q_new / jnp.sqrt(jnp.sum(q_new ** 2) + 1e-20)
        new_qpos = new_qpos.at[qa+3:qa+7].set(q_new)

    return new_qpos


# ── IK solver builder ────────────────────────────────────────────────

def build_ik_solver(m, site_ids, config=None):
    """Build a JIT-compiled batch IK solver using MJX.

    Two compilation modes:
    - scan mode (default): entire solve loop compiled via jax.lax.scan+vmap.
      Compiles slowly on CPU but runs fast on GPU. Best for large batches on GPU.
    - step mode (use_scan=False in config): only one gradient step is JIT-compiled.
      Python loop drives iterations. Compiles fast, good for CPU testing.

    Args:
        m: mujoco.MjModel (must have fix_geoms_for_mjx applied for MJX compat)
        site_ids: list of 24 site indices (from build_site_indices)
        config: IKConfig or None for defaults

    Returns:
        solve_batch: function (kp3d[B,24,3], valid[B,24]) -> (qpos[B,nq], residual_m[B])
        nq: number of qpos dimensions
    """
    enable_jax_cache()
    import jax
    import jax.numpy as jnp
    from mujoco import mjx

    if config is None:
        config = IKConfig()

    ji = build_joint_info(m)
    mx = mjx.put_model(m)
    si_j = jnp.array(site_ids)
    nq, nv = ji.nq, ji.nv

    max_iters = config.max_iters
    lr = config.lr
    beta = config.beta
    reg = config.reg
    use_scan = config.use_scan

    def fk_sites(mx, qpos):
        """Forward kinematics → all site positions."""
        dx = mjx.make_data(mx).replace(qpos=qpos)
        dx = mjx.kinematics(mx, dx)
        dx = mjx.com_pos(mx, dx)
        return dx.site_xpos

    def loss_from_vel(vel_step, qpos_current, kp3d, valid):
        """IK loss as a function of a velocity perturbation from current pose.

        The gradient of this w.r.t. vel_step lives in nv-space and correctly
        respects the quaternion manifold because it flows through integrate_pos_jax.
        """
        qpos_new = integrate_pos_jax(qpos_current, vel_step, 1.0, ji)
        sites = fk_sites(mx, qpos_new)
        kp_sites = sites[si_j]  # [24, 3]
        diff = (kp_sites - kp3d) * valid[:, None]  # [24, 3]
        ik_err = jnp.sum(diff ** 2)

        # L2 regularization on hinge joint angles
        hinge_vals = qpos_new[ji.hinge_qa]
        reg_loss = reg * jnp.sum(hinge_vals ** 2)

        return ik_err + reg_loss

    grad_fn = jax.grad(loss_from_vel, argnums=0)

    def _cold_start(kp3d, valid):
        """Initialize qpos with centroid alignment."""
        qpos = ji.qpos0.copy()
        if ji.free_joint_qa >= 0:
            qa = ji.free_joint_qa
            valid_mask = valid > 0.5
            n_valid = jnp.maximum(valid_mask.sum(), 1.0)
            centroid = jnp.where(valid_mask[:, None], kp3d, 0.0).sum(axis=0) / n_valid
            qpos = qpos.at[qa:qa+3].set(centroid)
        return qpos

    def _ik_step(qpos, momentum, kp3d, valid):
        """Single IK gradient step."""
        g = grad_fn(jnp.zeros(nv), qpos, kp3d, valid)
        momentum = beta * momentum + g
        qpos = integrate_pos_jax(qpos, -lr * momentum, 1.0, ji)
        qpos = jnp.clip(qpos, ji.clamp_lo, ji.clamp_hi)
        if ji.free_joint_qa >= 0:
            qa_ = ji.free_joint_qa
            q = qpos[qa_+3:qa_+7]
            q = q / jnp.sqrt(jnp.sum(q ** 2) + 1e-20)
            qpos = qpos.at[qa_+3:qa_+7].set(q)
        return qpos, momentum

    def _compute_residual(qpos, kp3d, valid):
        """Compute RMSE residual in meters."""
        sites = fk_sites(mx, qpos)
        kp_sites = sites[si_j]
        valid_mask = valid > 0.5
        diff_sq = jnp.sum((kp_sites - kp3d) ** 2, axis=-1)
        n_valid = jnp.maximum(valid_mask.sum(), 1.0)
        return jnp.sqrt(jnp.where(valid_mask, diff_sq, 0.0).sum() / n_valid)

    if use_scan:
        # Full scan mode: compile entire loop (slow compilation, fast on GPU)
        def solve_one(kp3d, valid):
            qpos = _cold_start(kp3d, valid)
            momentum = jnp.zeros(nv)

            def step(carry, _):
                qpos, momentum = carry
                qpos, momentum = _ik_step(qpos, momentum, kp3d, valid)
                return (qpos, momentum), None

            (qpos, _), _ = jax.lax.scan(step, (qpos, momentum), None, length=max_iters)
            rmse = _compute_residual(qpos, kp3d, valid)
            return qpos, rmse

        solve_batch = jax.jit(jax.vmap(solve_one))
    else:
        # Step mode: JIT single-frame functions, Python loops for iterations and frames.
        # Compiles fast (~10s) even on CPU. Processes frames sequentially.
        @jax.jit
        def single_step(qpos, momentum, kp3d, valid):
            """One gradient step for one frame."""
            return _ik_step(qpos, momentum, kp3d, valid)

        @jax.jit
        def single_init(kp3d, valid):
            """Cold-start one frame."""
            return _cold_start(kp3d, valid)

        @jax.jit
        def single_residual(qpos, kp3d, valid):
            """Compute residual for one frame."""
            return _compute_residual(qpos, kp3d, valid)

        def solve_batch(kp3d, valid):
            B = kp3d.shape[0]
            all_qpos = []
            all_res = []
            for b in range(B):
                qpos = single_init(kp3d[b], valid[b])
                momentum = jnp.zeros(nv)
                for _ in range(max_iters):
                    qpos, momentum = single_step(qpos, momentum, kp3d[b], valid[b])
                res = single_residual(qpos, kp3d[b], valid[b])
                all_qpos.append(qpos)
                all_res.append(res)
            return jnp.stack(all_qpos), jnp.stack(all_res)

    return solve_batch, nq


# ── High-level batch API ─────────────────────────────────────────────

def batch_ik_mjx(m, frames, site_ids, config=None):
    """Solve IK for a batch of frames on GPU using MJX.

    API matches batch_ik_cpu from ik_cpu.py for easy comparison.

    Args:
        m: mujoco.MjModel (will apply MJX geom fixes internally)
        frames: list of (kp3d[24,3], valid[24]) tuples
        site_ids: list of 24 site indices
        config: IKConfig or None for defaults

    Returns:
        all_qpos: [N, nq] numpy array
        residuals_mm: [N] numpy array (residuals in millimeters)
    """
    import jax.numpy as jnp
    import time

    if config is None:
        config = IKConfig()

    from .model import _fix_geoms_for_mjx

    # Apply MJX geom fixes (modifies model in place — caller should pass a dedicated copy)
    _fix_geoms_for_mjx(m)

    print(f"[ik_mjx] Building solver (max_iters={config.max_iters}, "
          f"lr={config.lr}, batch_size={config.batch_size})...")

    solve_batch, nq = build_ik_solver(m, site_ids, config)

    # Stack frames into arrays
    N = len(frames)
    kp3d_all = np.zeros((N, 24, 3), dtype=np.float32)
    valid_all = np.zeros((N, 24), dtype=np.float32)
    for i, (kp, v) in enumerate(frames):
        kp3d_all[i] = kp
        valid_all[i] = v

    all_qpos = np.zeros((N, nq), dtype=np.float64)
    all_residuals = np.zeros(N, dtype=np.float64)
    bs = config.batch_size

    # Warmup JIT compilation (must match batch_size for scan mode's static shapes)
    print(f"[ik_mjx] Compiling ({'scan' if config.use_scan else 'step'} mode)...")
    import jax
    t_compile = time.time()
    dummy_kp = jnp.zeros((bs, 24, 3))
    dummy_valid = jnp.ones((bs, 24))
    _ = solve_batch(dummy_kp, dummy_valid)
    jax.block_until_ready(_[0])
    print(f"[ik_mjx] Compiled in {time.time() - t_compile:.1f}s")

    # Process in batches
    t0 = time.time()
    n_batches = (N + bs - 1) // bs
    frames_done = 0

    for b in range(n_batches):
        start = b * bs
        end = min(start + bs, N)
        actual = end - start

        kp_batch = kp3d_all[start:end]
        valid_batch = valid_all[start:end]

        # Pad to batch_size for consistent JIT shapes
        if actual < bs:
            kp_batch = np.concatenate([kp_batch, np.zeros((bs - actual, 24, 3), dtype=np.float32)])
            valid_batch = np.concatenate([valid_batch, np.zeros((bs - actual, 24), dtype=np.float32)])

        qpos_batch, res_batch = solve_batch(jnp.array(kp_batch), jnp.array(valid_batch))
        jax.block_until_ready(qpos_batch)

        all_qpos[start:end] = np.array(qpos_batch[:actual])
        all_residuals[start:end] = np.array(res_batch[:actual]) * 1000.0  # m → mm

        frames_done += actual
        elapsed = time.time() - t0
        fps = frames_done / max(elapsed, 0.001)
        print(f"\r  [{frames_done}/{N} frames  {100*frames_done/N:.1f}%  {fps:.0f} fps]",
              end="", flush=True)

    elapsed = time.time() - t0
    print(f"\n[ik_mjx] Done: {N} frames in {elapsed:.1f}s "
          f"({N/max(elapsed, 0.001):.0f} fps)")

    # Statistics
    valid_mask = all_residuals >= 0
    if valid_mask.any():
        r = all_residuals[valid_mask]
        print(f"[ik_mjx] Residual: mean={r.mean():.2f}mm  "
              f"median={np.median(r):.2f}mm  p95={np.percentile(r, 95):.2f}mm")

    return all_qpos, all_residuals
