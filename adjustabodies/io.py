"""Data I/O: load keypoints, qpos exports, sessions, and fitted models."""

import os
import json
import math
import numpy as np
from typing import List, Tuple, Optional, Dict
from .arena import ArenaTransform


def load_keypoints3d(csv_path: str, max_frames: Optional[int] = None,
                      arena_tf: Optional[ArenaTransform] = None
                      ) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Load RED v2 keypoints3d.csv.

    Returns list of (kp3d[24,3], valid[24]) tuples.
    If arena_tf is provided, transforms keypoints to MuJoCo frame.
    """
    frames = []
    with open(csv_path) as f:
        for line in f:
            if line.startswith('#') or line.startswith('frame,'):
                continue
            parts = line.strip().split(',')
            kp = np.zeros((24, 3), dtype=np.float32)
            valid = np.zeros(24, dtype=np.float32)
            for k in range(24):
                base = 1 + k * 4
                if base + 2 < len(parts):
                    try:
                        x, y, z = float(parts[base]), float(parts[base+1]), float(parts[base+2])
                        if not (math.isnan(x) or math.isnan(y) or math.isnan(z)):
                            kp[k] = [x, y, z]
                            valid[k] = 1.0
                    except (ValueError, IndexError):
                        pass
            if arena_tf is not None:
                kp = arena_tf(kp).astype(np.float32)
            frames.append((kp, valid))
            if max_frames and len(frames) >= max_frames:
                break
    return frames


def load_qpos_export(csv_path: str, max_residual_mm: Optional[float] = None,
                      require_converged: bool = False
                      ) -> Tuple[Dict[int, np.ndarray], int]:
    """Load RED qpos_export.csv.

    Returns (dict[frame_id → qpos array], nq).
    """
    frames = {}
    nq = None
    with open(csv_path) as f:
        for line in f:
            line = line.strip()
            if line.startswith('# nq:'):
                nq = int(line.split(':')[1].strip())
                continue
            if not line or line.startswith('#'):
                continue
            if line.startswith('frame,'):
                cols = line.split(',')
                qpos_cols = [c for c in cols if c.startswith('qpos_')]
                if qpos_cols:
                    nq = len(qpos_cols)
                continue
            parts = line.split(',')
            frame_id = int(parts[0])
            if nq is None:
                nq = len(parts) - 4
            qpos = np.array([float(x) for x in parts[1:1+nq]], dtype=np.float32)
            residual = float(parts[1+nq])
            converged = bool(int(parts[3+nq]))
            if require_converged and not converged:
                continue
            if max_residual_mm is not None and residual > max_residual_mm:
                continue
            frames[frame_id] = qpos
    return frames, nq


def load_session(session_path: str) -> dict:
    """Load mujoco_session.json and return parsed contents."""
    with open(session_path) as f:
        return json.load(f)


def find_keypoints3d(project_dir: str) -> Optional[str]:
    """Find the latest keypoints3d.csv in a RED project directory."""
    labeled_dir = os.path.join(project_dir, 'labeled_data')
    if not os.path.isdir(labeled_dir):
        return None
    for session in sorted(os.listdir(labeled_dir), reverse=True):
        candidate = os.path.join(labeled_dir, session, 'keypoints3d.csv')
        if os.path.exists(candidate):
            return candidate
    return None


def save_fitted_model(m, output_path: str, metadata: dict = None):
    """Save a fitted MuJoCo model as .mjb with JSON sidecar."""
    import mujoco
    mujoco.mj_saveModel(m, output_path)

    if metadata:
        json_path = output_path.replace('.mjb', '.json')
        with open(json_path, 'w') as f:
            json.dump(metadata, f, indent=2)
