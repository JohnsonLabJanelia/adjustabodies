"""Rodent (rat/mouse) body model definitions.

Single source of truth for segment groups, site names, symmetry pairs,
and segment length initialization. Used by RED (C++ must sync), MJX
fitting, batch IK, and JARVIS-HybridNet qpos training.
"""

from dataclasses import dataclass, field
from typing import List, Tuple


@dataclass
class SpeciesConfig:
    """Configuration for a species' body model."""
    name: str
    sites: List[str]
    segments: List[Tuple[str, List[str]]]
    lr_site_pairs: List[Tuple[str, str]]
    midline_sites: List[str]
    segment_length_init: dict = field(default_factory=dict)


# ── 24 keypoint sites (Rat24Target skeleton order) ─────────────────────
RAT24_SITES = [
    "nose_0_kpsite",
    "ear_L_1_kpsite",
    "ear_R_2_kpsite",
    "neck_3_kpsite",
    "spineL_4_kpsite",
    "tailbase_5_kpsite",
    "shoulder_L_6_kpsite",
    "elbow_L_7_kpsite",
    "wrist_L_8_kpsite",
    "hand_L_9_kpsite",
    "shoulder_R_10_kpsite",
    "elbow_R_11_kpsite",
    "wrist_R_12_kpsite",
    "hand_R_13_kpsite",
    "knee_L_14_kpsite",
    "ankle_L_15_kpsite",
    "foot_L_16_kpsite",
    "knee_R_17_kpsite",
    "ankle_R_18_kpsite",
    "foot_R_19_kpsite",
    "tailtip_20_kpsite",
    "tailmid_21_kpsite",
    "tail1Q_22_kpsite",
    "tail3Q_23_kpsite",
]

# ── 12 L/R-symmetric segment groups ────────────────────────────────────
SEGMENT_DEFS = [
    ('head',      ['skull', 'jaw']),
    ('neck',      ['vertebra_cervical_5', 'vertebra_cervical_4', 'vertebra_cervical_3',
                   'vertebra_cervical_2', 'vertebra_cervical_1', 'vertebra_axis',
                   'vertebra_atlant']),
    ('spine',     [f'vertebra_{i}' for i in range(1, 7)]),
    ('pelvis',    ['pelvis']),
    ('tail',      [f'vertebra_C{i}' for i in range(1, 31)]),
    ('scapula',   ['scapula_L', 'scapula_R']),
    ('upper_arm', ['upper_arm_L', 'upper_arm_R']),
    ('lower_arm', ['lower_arm_L', 'lower_arm_R']),
    ('hand',      ['hand_L', 'hand_R', 'finger_L', 'finger_R']),
    ('upper_leg', ['upper_leg_L', 'upper_leg_R']),
    ('lower_leg', ['lower_leg_L', 'lower_leg_R']),
    ('foot',      ['foot_L', 'foot_R', 'toe_L', 'toe_R']),
]

# ── L/R symmetry pairs (site names) ───────────────────────────────────
LR_SITE_PAIRS = [
    ("ear_L_1_kpsite",      "ear_R_2_kpsite"),
    ("shoulder_L_6_kpsite", "shoulder_R_10_kpsite"),
    ("elbow_L_7_kpsite",    "elbow_R_11_kpsite"),
    ("wrist_L_8_kpsite",    "wrist_R_12_kpsite"),
    ("hand_L_9_kpsite",     "hand_R_13_kpsite"),
    ("knee_L_14_kpsite",    "knee_R_17_kpsite"),
    ("ankle_L_15_kpsite",   "ankle_R_18_kpsite"),
    ("foot_L_16_kpsite",    "foot_R_19_kpsite"),
]

# ── Midline sites (Y offset = 0 under symmetry) ───────────────────────
MIDLINE_SITES = [
    "nose_0_kpsite", "neck_3_kpsite", "spineL_4_kpsite", "tailbase_5_kpsite",
    "tailtip_20_kpsite", "tailmid_21_kpsite", "tail1Q_22_kpsite", "tail3Q_23_kpsite",
]

# ── Segment length initialization (from data analysis) ─────────────────
# Ratios of data segment lengths to model segment lengths.
# Used to seed the optimizer (MJX gradient descent refines from here).
SEGMENT_LENGTH_INIT = {
    'head': 0.92, 'neck': 1.0, 'spine': 1.26, 'pelvis': 1.0,
    'tail': 1.0, 'scapula': 1.0, 'upper_arm': 0.69, 'lower_arm': 0.86,
    'hand': 0.9, 'upper_leg': 0.9, 'lower_leg': 1.19, 'foot': 1.0,
}

# ── Skeleton node names (RED's display names → site mapping) ──────────
# Maps RED skeleton node names to MuJoCo site names.
SKELETON_TO_SITE = {
    "Snout":     "nose_0_kpsite",
    "EarL":      "ear_L_1_kpsite",
    "EarR":      "ear_R_2_kpsite",
    "Neck":      "neck_3_kpsite",
    "SpineL":    "spineL_4_kpsite",
    "TailBase":  "tailbase_5_kpsite",
    "ShoulderL": "shoulder_L_6_kpsite",
    "ElbowL":    "elbow_L_7_kpsite",
    "WristL":    "wrist_L_8_kpsite",
    "HandL":     "hand_L_9_kpsite",
    "ShoulderR": "shoulder_R_10_kpsite",
    "ElbowR":    "elbow_R_11_kpsite",
    "WristR":    "wrist_R_12_kpsite",
    "HandR":     "hand_R_13_kpsite",
    "KneeL":     "knee_L_14_kpsite",
    "AnkleL":    "ankle_L_15_kpsite",
    "FootL":     "foot_L_16_kpsite",
    "KneeR":     "knee_R_17_kpsite",
    "AnkleR":    "ankle_R_18_kpsite",
    "FootR":     "foot_R_19_kpsite",
    "TailTip":   "tailtip_20_kpsite",
    "TailMid":   "tailmid_21_kpsite",
    "Tail1Q":    "tail1Q_22_kpsite",
    "Tail3Q":    "tail3Q_23_kpsite",
}

# ── Assembled config ───────────────────────────────────────────────────
RODENT_CONFIG = SpeciesConfig(
    name="rodent",
    sites=RAT24_SITES,
    segments=SEGMENT_DEFS,
    lr_site_pairs=LR_SITE_PAIRS,
    midline_sites=MIDLINE_SITES,
    segment_length_init=SEGMENT_LENGTH_INIT,
)
