"""CPU-based IK solver using MuJoCo C API.

Gradient descent with momentum, matching RED's mujoco_ik.h algorithm.
Used for Q-phase in alternating optimization and for batch IK export.
"""

import numpy as np
import mujoco
from typing import List, Tuple


def solve_ik_frame(m: mujoco.MjModel, d: mujoco.MjData,
                    targets: List[Tuple[int, np.ndarray]],
                    max_iters: int = 1000, lr: float = 0.001,
                    beta: float = 0.99, warm_start: bool = False):
    """Solve IK for a single frame.

    Args:
        m: MuJoCo model
        d: MuJoCo data (qpos will be modified in place)
        targets: list of (site_id, target_xyz) pairs
        max_iters: gradient descent iterations
        lr: learning rate
        beta: momentum coefficient
        warm_start: if False, reset qpos and align root to target centroid
    """
    nv = m.nv
    jacp = np.zeros((3, nv))
    update = np.zeros(nv)

    if not warm_start:
        mujoco.mj_resetData(m, d)
        # Root alignment: center model on target centroid
        if targets:
            centroid = np.mean([t[1] for t in targets], axis=0)
            for j in range(m.njnt):
                if m.jnt_type[j] == 0:  # FREE joint
                    qa = int(m.jnt_qposadr[j])
                    d.qpos[qa:qa+3] = centroid
                    d.qpos[qa+3] = 1.0  # quaternion w
                    break
        mujoco.mj_forward(m, d)

    for _ in range(max_iters):
        grad = np.zeros(nv)
        for sid, tgt in targets:
            jacp[:] = 0
            mujoco.mj_jacSite(m, d, jacp, None, sid)
            grad += 2.0 * jacp.T @ (d.site_xpos[sid] - tgt)
        update = beta * update + grad
        step = -lr * update
        mujoco.mj_integratePos(m, d.qpos, step, 1.0)
        mujoco.mj_fwdPosition(m, d)

    return d.qpos.copy()


def batch_ik_cpu(m: mujoco.MjModel, frames, site_ids: List[int],
                  max_iters: int = 1000, **kwargs) -> np.ndarray:
    """Solve IK for a batch of frames on CPU (no warm-start between frames).

    Args:
        m: MuJoCo model
        frames: list of (kp3d[24,3], valid[24]) tuples
        site_ids: list of site indices for the 24 keypoints
        max_iters: IK iterations per frame

    Returns:
        all_qpos: [N, nq] array of solved joint angles
    """
    d = mujoco.MjData(m)
    N = len(frames)
    all_qpos = np.zeros((N, m.nq), dtype=np.float64)

    for i, (kp_mj, valid) in enumerate(frames):
        targets = [(site_ids[k], kp_mj[k])
                    for k in range(24) if valid[k] > 0.5 and site_ids[k] >= 0]
        if not targets:
            continue
        all_qpos[i] = solve_ik_frame(m, d, targets, max_iters=max_iters, **kwargs)

    return all_qpos


def batch_ik_cpu_trial(m: mujoco.MjModel, frames, site_ids: List[int],
                        max_iters: int = 1000, warm_iters: int = 200,
                        **kwargs) -> Tuple[np.ndarray, np.ndarray]:
    """Solve IK for a sequence of frames with warm-starting (one trial).

    Frame 0 cold-starts, subsequent frames warm-start from the previous
    solution with fewer iterations. This produces temporally consistent
    poses essential for UMAP and behavioral analysis.

    Args:
        m: MuJoCo model
        frames: list of (kp3d[24,3], valid[24]) tuples (sequential)
        site_ids: list of site indices
        max_iters: iterations for cold-start (frame 0)
        warm_iters: iterations for warm-start (frames 1+)

    Returns:
        all_qpos: [N, nq] array
        residuals: [N] array in meters
    """
    d = mujoco.MjData(m)
    N = len(frames)
    all_qpos = np.zeros((N, m.nq), dtype=np.float64)
    residuals = np.full(N, np.nan, dtype=np.float64)

    for i, (kp_mj, valid) in enumerate(frames):
        targets = [(site_ids[k], kp_mj[k])
                    for k in range(24) if valid[k] > 0.5 and site_ids[k] >= 0]
        if not targets:
            # No valid keypoints — reset warm-start
            mujoco.mj_resetData(m, d)
            continue

        iters = max_iters if i == 0 else warm_iters
        ws = (i > 0)  # warm-start from previous frame's qpos (still in d)

        all_qpos[i] = solve_ik_frame(m, d, targets, max_iters=iters,
                                      warm_start=ws, **kwargs)

        # Compute residual
        err_sq = sum(np.sum((d.site_xpos[sid] - tgt) ** 2)
                     for sid, tgt in targets)
        residuals[i] = np.sqrt(err_sq / len(targets))

    return all_qpos, residuals
