"""Measure segment lengths from 3D keypoint data.

Computes median inter-keypoint distances with strict outlier rejection.
Used to constrain body model fitting (prevents tail inflation etc.).
"""

import numpy as np
from typing import List, Tuple, Dict


# Segment chains defined by keypoint index pairs
# These are the segments we can directly measure from labeled data
MEASURABLE_SEGMENTS = {
    # Tail chain: TailBase(5) → Tail1Q(22) → TailMid(21) → Tail3Q(23) → TailTip(20)
    'tail_base_1q': (5, 22),
    'tail_1q_mid': (22, 21),
    'tail_mid_3q': (21, 23),
    'tail_3q_tip': (23, 20),
    # Spine
    'neck_spineL': (3, 4),
    'spineL_tailbase': (4, 5),
    # Head
    'snout_neck': (0, 3),
    'ear_span': (1, 2),
    # Upper arms
    'shoulder_elbow_L': (6, 7),
    'shoulder_elbow_R': (10, 11),
    # Lower arms
    'elbow_wrist_L': (7, 8),
    'elbow_wrist_R': (11, 12),
    # Upper legs
    'knee_ankle_L': (14, 15),
    'knee_ankle_R': (17, 18),
    # Lower legs
    'ankle_foot_L': (15, 16),
    'ankle_foot_R': (18, 19),
}

# Map segment measurement names to body model segment names (for constraining fitting)
SEGMENT_TO_MODEL = {
    'tail_base_1q': 'tail',
    'tail_1q_mid': 'tail',
    'tail_mid_3q': 'tail',
    'tail_3q_tip': 'tail',
    'neck_spineL': 'spine',
    'spineL_tailbase': 'spine',
    'snout_neck': 'head',
    'shoulder_elbow_L': 'upper_arm',
    'shoulder_elbow_R': 'upper_arm',
    'elbow_wrist_L': 'lower_arm',
    'elbow_wrist_R': 'lower_arm',
    'knee_ankle_L': 'upper_leg',
    'knee_ankle_R': 'upper_leg',
    'ankle_foot_L': 'lower_leg',
    'ankle_foot_R': 'lower_leg',
}


def measure_segment_lengths(frames: List[Tuple[np.ndarray, np.ndarray]],
                             outlier_threshold: float = 0.15
                             ) -> Dict[str, float]:
    """Measure median segment lengths from 3D keypoint frames.

    Uses strict outlier rejection: only keeps frames where ALL segments
    of a chain are within threshold of their median.

    Args:
        frames: list of (kp3d[24,3], valid[24]) tuples (in meters)
        outlier_threshold: fraction of median for rejection (0.15 = ±15%)

    Returns:
        dict mapping segment name → median length in meters
    """
    # Collect per-segment lengths
    all_lengths = {name: [] for name in MEASURABLE_SEGMENTS}

    for kp, valid in frames:
        for name, (a, b) in MEASURABLE_SEGMENTS.items():
            if valid[a] > 0.5 and valid[b] > 0.5:
                length = np.linalg.norm(kp[b] - kp[a])
                if np.isfinite(length) and length > 0.001:  # > 1mm
                    all_lengths[name].append(length)

    # Compute medians
    medians = {}
    for name in MEASURABLE_SEGMENTS:
        if len(all_lengths[name]) > 10:
            medians[name] = np.median(all_lengths[name])

    # Strict outlier rejection per chain group
    # For tail: reject frames where any tail segment is > threshold from median
    chain_groups = {
        'tail': ['tail_base_1q', 'tail_1q_mid', 'tail_mid_3q', 'tail_3q_tip'],
        'arm_L': ['shoulder_elbow_L', 'elbow_wrist_L'],
        'arm_R': ['shoulder_elbow_R', 'elbow_wrist_R'],
        'leg_L': ['knee_ankle_L', 'ankle_foot_L'],
        'leg_R': ['knee_ankle_R', 'ankle_foot_R'],
    }

    filtered_lengths = {name: [] for name in MEASURABLE_SEGMENTS}

    for kp, valid in frames:
        for group_name, seg_names in chain_groups.items():
            # Check all segments in this chain are valid and near median
            all_ok = True
            lengths = {}
            for name in seg_names:
                a, b = MEASURABLE_SEGMENTS[name]
                if valid[a] <= 0.5 or valid[b] <= 0.5:
                    all_ok = False; break
                length = np.linalg.norm(kp[b] - kp[a])
                if not np.isfinite(length) or length < 0.001:
                    all_ok = False; break
                if name in medians:
                    if abs(length - medians[name]) / medians[name] > outlier_threshold:
                        all_ok = False; break
                lengths[name] = length

            if all_ok:
                for name, length in lengths.items():
                    filtered_lengths[name].append(length)

        # Non-chain segments (spine, head, ear)
        for name in ['neck_spineL', 'spineL_tailbase', 'snout_neck', 'ear_span']:
            a, b = MEASURABLE_SEGMENTS[name]
            if valid[a] > 0.5 and valid[b] > 0.5:
                length = np.linalg.norm(kp[b] - kp[a])
                if np.isfinite(length) and length > 0.001:
                    if name in medians:
                        if abs(length - medians[name]) / medians[name] <= outlier_threshold:
                            filtered_lengths[name].append(length)

    # Final medians from filtered data
    result = {}
    for name in MEASURABLE_SEGMENTS:
        if len(filtered_lengths[name]) > 5:
            result[name] = float(np.median(filtered_lengths[name]))

    return result


def compute_model_segment_scale(measured: Dict[str, float],
                                 model_lengths: Dict[str, float]
                                 ) -> Dict[str, float]:
    """Compute per-segment scale factors from measured vs model lengths.

    For segments that map to the same model segment (e.g., all 4 tail segments
    map to 'tail'), computes the median scale across all measurements.

    Args:
        measured: from measure_segment_lengths (meters)
        model_lengths: from measure_model_segments (meters)

    Returns:
        dict mapping model segment name → target scale factor
    """
    segment_scales = {}
    for name in measured:
        model_seg = SEGMENT_TO_MODEL.get(name)
        if model_seg and name in model_lengths and model_lengths[name] > 0.001:
            scale = measured[name] / model_lengths[name]
            if model_seg not in segment_scales:
                segment_scales[model_seg] = []
            segment_scales[model_seg].append(scale)

    # Take median scale per model segment
    result = {}
    for seg, scales in segment_scales.items():
        result[seg] = float(np.median(scales))

    return result


def measure_model_segments(m, site_ids: List[int]) -> Dict[str, float]:
    """Measure segment lengths from a MuJoCo model in its default pose.

    Args:
        m: mujoco.MjModel
        site_ids: list of 24 site indices

    Returns:
        dict mapping segment name → length in meters
    """
    import mujoco
    d = mujoco.MjData(m)
    mujoco.mj_resetData(m, d)
    mujoco.mj_forward(m, d)

    result = {}
    for name, (a, b) in MEASURABLE_SEGMENTS.items():
        if site_ids[a] >= 0 and site_ids[b] >= 0:
            sa = d.site_xpos[site_ids[a]]
            sb = d.site_xpos[site_ids[b]]
            result[name] = float(np.linalg.norm(sb - sa))

    return result
