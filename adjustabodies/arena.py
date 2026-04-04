"""Arena coordinate transform: calibration frame (mm) → MuJoCo frame (meters).

The arena alignment is a rigid transform computed from arena corner
correspondences: p_mj = scale * R @ p_calib + t

Stored in mujoco_session.json under the 'arena' key.
"""

import json
import numpy as np


class ArenaTransform:
    """Rigid transform from calibration frame (mm) to MuJoCo frame (meters)."""

    def __init__(self, R=None, t=None, scale=0.001):
        self.R = R if R is not None else np.eye(3)
        self.t = t if t is not None else np.zeros(3)
        self.scale = scale

    def __call__(self, p_mm):
        """Transform points: [N, 3] or [3] in mm → meters."""
        return self.scale * (p_mm @ self.R.T) + self.t

    def inverse(self, p_mj):
        """Inverse: MuJoCo meters → calibration mm."""
        return ((p_mj - self.t) @ self.R) / self.scale

    @classmethod
    def from_session(cls, session_path):
        """Load from mujoco_session.json."""
        with open(session_path) as f:
            session = json.load(f)
        arena = session.get('arena', {})
        if not arena.get('valid', False):
            return cls()
        R_flat = arena.get('R', [1, 0, 0, 0, 1, 0, 0, 0, 1])
        R = np.array(R_flat, dtype=np.float64).reshape(3, 3)
        t = np.array(arena.get('t', [0, 0, 0]), dtype=np.float64)
        scale = arena.get('scale', 0.001)
        return cls(R=R, t=t, scale=scale)
