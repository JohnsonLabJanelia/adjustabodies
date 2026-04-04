"""Test species definitions are consistent and complete."""
from adjustabodies.species.rodent import (
    RAT24_SITES, SEGMENT_DEFS, LR_SITE_PAIRS, MIDLINE_SITES,
    SKELETON_TO_SITE, RODENT_CONFIG,
)


def test_rat24_sites_count():
    assert len(RAT24_SITES) == 24


def test_segment_defs_cover_key_bodies():
    all_bodies = [b for _, bodies in SEGMENT_DEFS for b in bodies]
    assert 'skull' in all_bodies
    assert 'pelvis' in all_bodies
    assert 'upper_arm_L' in all_bodies
    assert 'upper_arm_R' in all_bodies


def test_lr_pairs_symmetric():
    for l, r in LR_SITE_PAIRS:
        assert '_L_' in l or '_L' in l.split('_kpsite')[0]
        assert '_R_' in r or '_R' in r.split('_kpsite')[0]


def test_midline_no_lr():
    for s in MIDLINE_SITES:
        assert '_L_' not in s and '_R_' not in s


def test_skeleton_to_site_complete():
    assert len(SKELETON_TO_SITE) == 24
    assert set(SKELETON_TO_SITE.values()) == set(RAT24_SITES)


def test_config_assembled():
    assert RODENT_CONFIG.name == "rodent"
    assert len(RODENT_CONFIG.sites) == 24
    assert len(RODENT_CONFIG.segments) == 12
