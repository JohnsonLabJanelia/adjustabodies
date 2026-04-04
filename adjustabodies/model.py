"""MuJoCo model loading, scaling, and MJX compatibility fixes."""

import numpy as np
import mujoco
from typing import List, Tuple, Optional, Dict
from .species.rodent import SEGMENT_DEFS, RAT24_SITES


def load_model(xml_path: str, add_free_joint: bool = True,
               fix_geoms_for_mjx: bool = False) -> mujoco.MjModel:
    """Load a MuJoCo model from XML, optionally adding free joint and fixing geoms.

    Args:
        xml_path: Path to .xml or .mjb file
        add_free_joint: Add free joint to torso/thorax body (for IK)
        fix_geoms_for_mjx: Convert ellipsoid→sphere, disable box collision (MJX compat)
    """
    if xml_path.endswith('.mjb'):
        m = mujoco.MjModel.from_binary_path(xml_path)
    else:
        spec = mujoco.MjSpec.from_file(xml_path)
        if add_free_joint:
            for root_name in ["torso", "thorax"]:
                body = spec.body(root_name)
                if body is not None:
                    body.add_freejoint()
                    break
        m = spec.compile()

    if fix_geoms_for_mjx:
        _fix_geoms_for_mjx(m)

    return m


def _fix_geoms_for_mjx(m: mujoco.MjModel):
    """Fix geom types for MJX compatibility (FK only, no collision physics)."""
    for i in range(m.ngeom):
        if m.geom_type[i] == 8:  # ELLIPSOID → SPHERE
            m.geom_type[i] = 2
        if m.geom_type[i] == 6:  # BOX → disable collision
            m.geom_contype[i] = 0
            m.geom_conaffinity[i] = 0
    m.opt.disableflags |= (mujoco.mjtDisableBit.mjDSBL_CONTACT |
                            mujoco.mjtDisableBit.mjDSBL_CONSTRAINT)


def build_segment_indices(m: mujoco.MjModel,
                           segment_defs: List[Tuple[str, List[str]]] = None
                           ) -> List[Tuple[str, List[int]]]:
    """Map segment names → body IDs for a compiled model."""
    if segment_defs is None:
        segment_defs = SEGMENT_DEFS
    segments = []
    for name, body_names in segment_defs:
        bids = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, bn)
                for bn in body_names]
        bids = [b for b in bids if b >= 0]
        if bids:
            segments.append((name, bids))
    return segments


def build_site_indices(m: mujoco.MjModel,
                       site_names: List[str] = None) -> List[int]:
    """Map site names → model site IDs. Returns -1 for missing sites."""
    if site_names is None:
        site_names = RAT24_SITES
    site_ids = []
    for name in site_names:
        sid = -1
        for i in range(m.nsite):
            if m.site(i).name == name:
                sid = i
                break
        site_ids.append(sid)
    return site_ids


def apply_segment_scales(m: mujoco.MjModel,
                          segments: List[Tuple[str, List[int]]],
                          scales: Dict[str, float],
                          originals: Optional[Dict[str, np.ndarray]] = None):
    """Apply per-segment scale factors to a compiled model.

    If originals is provided, restores to original values first (avoids compounding).
    """
    if originals:
        for k, v in originals.items():
            getattr(m, k)[:] = v

    for seg_name, bids in segments:
        s = scales.get(seg_name, 1.0)
        if abs(s - 1.0) < 1e-8:
            continue
        s3, s5 = s ** 3, s ** 5
        for bid in bids:
            m.body_pos[bid] *= s
            m.body_ipos[bid] *= s
            m.body_mass[bid] *= s3
            m.body_inertia[bid] *= s5
            for gi in range(m.ngeom):
                if m.geom_bodyid[gi] == bid:
                    m.geom_pos[gi] *= s
                    m.geom_size[gi] *= s
            for si in range(m.nsite):
                if m.site_bodyid[si] == bid:
                    m.site_pos[si] *= s
            for ji in range(m.njnt):
                if m.jnt_bodyid[ji] == bid:
                    m.jnt_pos[ji] *= s

    mujoco.mj_setConst(m, mujoco.MjData(m))


def save_originals(m: mujoco.MjModel) -> Dict[str, np.ndarray]:
    """Snapshot model arrays for later restore."""
    return {k: getattr(m, k).copy() for k in
            ['body_pos', 'body_ipos', 'geom_pos', 'geom_size',
             'site_pos', 'jnt_pos', 'body_mass', 'body_inertia']}
