"""Compute generalized velocities (qvel) from qpos time series.

Uses mujoco.mj_differentiatePos for correct quaternion handling,
with optional Savitzky-Golay smoothing.
"""

import numpy as np
import mujoco
from typing import Optional


def compute_qvel(m: mujoco.MjModel, qpos_series: np.ndarray,
                 fps: float = 180.0,
                 smooth_window: int = 0,
                 smooth_polyorder: int = 2) -> np.ndarray:
    """Compute qvel from a time series of qpos.

    Uses mj_differentiatePos which correctly handles quaternion→angular
    velocity via the exponential map. For hinge joints this is simply
    finite differencing; for the free joint quaternion it computes
    body-frame angular velocity.

    Args:
        m: MuJoCo model
        qpos_series: [T, nq] array of joint positions over time
        fps: frame rate (default 180)
        smooth_window: Savitzky-Golay window size in frames (0 = no smoothing).
                       Must be odd. Recommended: 37 for 180fps (≈0.2s).
        smooth_polyorder: SG polynomial order (default 2)

    Returns:
        [T, nv] array of generalized velocities. First frame is zero
        (no previous frame to differentiate against).
    """
    T, nq = qpos_series.shape
    nv = m.nv
    dt = 1.0 / fps

    qvel = np.zeros((T, nv), dtype=np.float64)

    # Differentiate consecutive frames
    for t in range(1, T):
        mujoco.mj_differentiatePos(m, qvel[t], dt, qpos_series[t-1], qpos_series[t])

    # First frame: use forward difference (same as frame 1, but computed from 0→1)
    # This avoids a zero spike without duplicating frame 1's velocity.
    # Already computed: qvel[1] = diff(qpos[0], qpos[1]) / dt, so qvel[0] = qvel[1]
    # is the best we can do without a frame before frame 0.
    if T > 1:
        qvel[0] = qvel[1]  # forward difference = backward difference at frame 1

    # Optional Savitzky-Golay smoothing
    if smooth_window > 0 and T > smooth_window:
        from scipy.signal import savgol_filter
        if smooth_window % 2 == 0:
            smooth_window += 1  # must be odd
        qvel = savgol_filter(qvel, smooth_window, smooth_polyorder, axis=0)

    return qvel


def extract_hinge_features(qpos_series: np.ndarray, qvel_series: np.ndarray,
                            m: mujoco.MjModel) -> dict:
    """Extract hinge joint angles and velocities, dropping the free joint.

    Args:
        qpos_series: [T, nq]
        qvel_series: [T, nv]
        m: MuJoCo model

    Returns:
        dict with keys 'qpos_hinges' [T, n_hinge], 'qvel_hinges' [T, n_hinge],
        'hinge_names' list, 'com_speed' [T] (translational speed in m/s)
    """
    hinge_qa = []
    hinge_va = []
    hinge_names = []
    free_va = -1

    for j in range(m.njnt):
        if m.jnt_type[j] == 0:  # FREE
            free_va = int(m.jnt_dofadr[j])
        elif m.jnt_type[j] in (2, 3):  # SLIDE, HINGE
            hinge_qa.append(int(m.jnt_qposadr[j]))
            hinge_va.append(int(m.jnt_dofadr[j]))
            hinge_names.append(m.joint(j).name)

    qpos_hinges = qpos_series[:, hinge_qa]
    qvel_hinges = qvel_series[:, hinge_va]

    # COM speed: norm of free joint translational velocity
    com_speed = np.zeros(len(qvel_series))
    if free_va >= 0:
        com_speed = np.linalg.norm(qvel_series[:, free_va:free_va+3], axis=1)

    return {
        'qpos_hinges': qpos_hinges.astype(np.float32),
        'qvel_hinges': qvel_hinges.astype(np.float32),
        'hinge_names': hinge_names,
        'com_speed': com_speed.astype(np.float32),
    }
