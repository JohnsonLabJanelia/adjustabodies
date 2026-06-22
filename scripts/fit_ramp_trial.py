#!/usr/bin/env python3
"""Fit a body model from ONE ramp_curr trial window and export a portable XML.

Picks the first valid+success trial for a rat from dataset_meta.csv, slices that
session's keypoints3d.csv over the absolute frame window
[trial_start+window_start .. trial_start+<end_col>], runs the adjustabodies
2-phase fit (scripts.fit_green_rats.fit_on_frames), saves the fitted .mjb, and
transplants it onto the base XML for a self-contained .xml.

Keypoints in the per-session CSV are `frame,(id,x,y,z)*25` in arena mm, id order
== RAT24_SITES (id 24 is the ball, dropped). Only mm->m (*0.001) is applied: the
fit is invariant to per-frame rigid transforms because the root free joint
absorbs them (same convention as fit_green_rats.extract_frames).
"""
import argparse
import csv
import importlib.util
import os
import sys

import numpy as np

ADJ = os.path.expanduser("~/src/adjustabodies")
sys.path.insert(0, ADJ)
from scripts.fit_green_rats import fit_on_frames  # noqa: E402

RAMP_DIR = os.path.expanduser("~/dataset/ramp_curr_v3")
BASE_XML = os.path.join(ADJ, "models", "rodent_data_driven_limits.xml")
MAKE_XML = os.path.expanduser(
    "~/src/fetch_analysis_examples/figure1/11_ik_rollout_adjustbody/make_body_xml.py")


def pick_trials(meta_csv, rat, n):
    """First n valid+success trials for a rat -> list of (row_idx, row)."""
    rows = list(csv.DictReader(open(meta_csv)))
    out = []
    for i, r in enumerate(rows):
        if (r["animal"] == rat and r["is_valid"] == "True"
                and r["is_success"] == "True"):
            out.append((i, r))
            if len(out) >= n:
                break
    if not out:
        raise SystemExit(f"no valid+success trial for {rat}")
    return out


def _parse_row(v):
    """One keypoints3d data row (after frame col) -> (kp[24,3] m, valid[24])."""
    pts = np.array(
        [[float(v[i + 1]), float(v[i + 2]), float(v[i + 3])]
         for i in range(0, len(v) - 3, 4)], dtype=np.float32)  # (25,3) mm
    kp = pts[:24] * 0.001  # mm -> m, drop ball (id 24)
    valid = np.isfinite(kp).all(axis=-1) & ~(kp == 0).all(axis=-1)
    return kp.astype(np.float32), valid.astype(np.float32)


def read_windows(kp_csv, windows):
    """windows: list of (lo, hi) abs-frame ranges. Reads the session CSV once and
    returns frames whose abs index falls in ANY window -> list of (kp, valid)."""
    frames = []
    with open(kp_csv) as f:
        rd = csv.reader(f)
        next(rd)  # "Rat24Target" title line
        for row in rd:
            if not row:
                continue
            fr = int(float(row[0]))
            if any(lo <= fr <= hi for lo, hi in windows):
                frames.append(_parse_row(row[1:]))
    return frames


def fit_phase1(model_xml, frames, output_mjb, label,
               n_rounds=3, m_iters=100, ik_iters=500, target_weight=0.05,
               freeze_unmeasured=False):
    """Phase 1 only (segment scaling) -> saved .mjb. Mirrors fit_on_frames but
    stops after apply_segment_scales (no STAC site offsets).

    freeze_unmeasured: if True, pin segments that no keypoint pair can measure
        (scapula, neck, pelvis, hand, foot) at rel-scale 1.0 -- only the 7
        keypoint-measured segments are optimized."""
    import time
    import mujoco
    from mujoco import mjx
    from adjustabodies import enable_jax_cache
    enable_jax_cache()
    from adjustabodies.model import (load_model, build_segment_indices,
                                     build_site_indices, save_originals,
                                     apply_segment_scales)
    from adjustabodies.io import save_fitted_model
    from adjustabodies.resize import run_resize_phase, build_mjx_scale_fn
    from adjustabodies.segment_lengths import (
        measure_segment_lengths, measure_model_segments, compute_model_segment_scale,
        SEGMENT_TO_MODEL)
    from adjustabodies.species.rodent import SEGMENT_DEFS, RAT24_SITES, SEGMENT_LENGTH_INIT

    t0 = time.time()
    m = load_model(model_xml, add_free_joint=True, fix_geoms_for_mjx=True)
    segments = build_segment_indices(m, SEGMENT_DEFS)
    site_ids = build_site_indices(m, RAT24_SITES)
    orig = save_originals(m)
    mx_base = mjx.put_model(m)
    apply_scales_fn = build_mjx_scale_fn(m, segments, orig)
    init_rel = np.array([SEGMENT_LENGTH_INIT.get(n, 1.0) for n, _ in segments],
                        dtype=np.float32)

    # Segments with no keypoint span -> optionally frozen at rel 1.0.
    measurable = set(SEGMENT_TO_MODEL.values())
    freeze_rel = None
    if freeze_unmeasured:
        freeze_rel = np.array([name not in measurable for name, _ in segments])
        frozen = [name for (name, _), f in zip(segments, freeze_rel) if f]
        print(f"  freezing unmeasurable segments at rel 1.0: {frozen}")

    measured = measure_segment_lengths(frames)
    model_lengths = measure_model_segments(m, site_ids)
    targets = compute_model_segment_scale(measured, model_lengths)

    params, pre_res, post_res = run_resize_phase(
        m, mx_base, segments, site_ids, orig, frames, apply_scales_fn,
        init_global=1.0, init_rel_scales=init_rel,
        n_rounds=n_rounds, m_iters=m_iters, ik_iters=ik_iters,
        lr_scale=0.003, reg_scale=0.001,
        segment_targets=targets, segment_target_weight=target_weight,
        freeze_rel=freeze_rel, verbose=True)

    gs = float(params['global_scale'])
    rs = np.array(params['rel_scales'])
    scales = {name: gs * rs[g] for g, (name, _) in enumerate(segments)}
    apply_segment_scales(m, segments, scales, orig)
    mujoco.mj_setConst(m, mujoco.MjData(m))

    meta = {'adjustabodies_version': '0.1.0', 'base_model': model_xml,
            'data_source': label, 'n_frames': len(frames), 'phases_run': 1,
            'phase1_residual_mm': post_res, 'segment_scales': scales}
    save_fitted_model(m, output_mjb, meta)
    print(f"\nPhase-1 fit: residual {pre_res:.2f} -> {post_res:.2f} mm  "
          f"({time.time()-t0:.0f}s)\n  saved {output_mjb}")
    for name, s in scales.items():
        print(f"  {name:<12s} {s:.3f}")
    return scales


def fix_ankle_range(xml_path):
    """The base model's data-driven ankle_L limit [0.0523, ...] excludes the
    neutral pose (0), so a free-sim starts out-of-range and the foot vibrates.
    Set ankle_L's lower bound to 0 in the exported XML."""
    import re
    txt = open(xml_path).read()
    new, n = re.subn(r'(<joint name="ankle_L"[^>]*range=")[^"]*(")',
                     r'\g<1>0 1.3963\g<2>', txt)
    if n:
        open(xml_path, "w").write(new)
    print(f"  ankle_L range -> [0, 1.3963] ({n} joint patched)")


def convert_fixed_base(mjb, base_xml, out):
    """Transplant fitted geometry from a .mjb onto the base spec WITHOUT adding a
    root free joint -> torso welded to world (won't drop in the viewer)."""
    import mujoco
    me = mujoco.MjModel.from_binary_path(mjb)
    spec = mujoco.MjSpec.from_file(base_xml)   # NO add_freejoint
    B, G, S, J = (mujoco.mjtObj.mjOBJ_BODY, mujoco.mjtObj.mjOBJ_GEOM,
                  mujoco.mjtObj.mjOBJ_SITE, mujoco.mjtObj.mjOBJ_JOINT)
    nid = lambda o, n: mujoco.mj_name2id(me, o, n)
    for b in spec.bodies:
        if not b.name or b.name == "world":
            continue
        i = nid(B, b.name); b.pos = me.body_pos[i].copy(); b.quat = me.body_quat[i].copy()
    for g in spec.geoms:
        if not g.name:
            continue
        i = nid(G, g.name)
        g.pos = me.geom_pos[i].copy(); g.quat = me.geom_quat[i].copy(); g.size = me.geom_size[i].copy()
    for s in spec.sites:
        if not s.name:
            continue
        i = nid(S, s.name); s.pos = me.site_pos[i].copy(); s.quat = me.site_quat[i].copy()
    for j in spec.joints:
        if not j.name:
            continue
        i = nid(J, j.name)
        if i >= 0:
            j.pos = me.jnt_pos[i].copy()
    open(out, "w").write(spec.to_xml())
    print(f"  wrote fixed-base {out}")


def load_make_xml():
    spec = importlib.util.spec_from_file_location("make_body_xml", MAKE_XML)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rat", default="emilie")
    ap.add_argument("--n-trials", type=int, default=10,
                    help="pool the first N valid+success trials for this rat.")
    ap.add_argument("--max-frames", type=int, default=300,
                    help="subsample pooled frames to at most this many "
                         "(even-spaced; adjacent frames are highly redundant).")
    ap.add_argument("--end-col", default="first_contact",
                    help="trial-relative column for the window end (the 'catch').")
    ap.add_argument("--out-dir", default=os.path.expanduser(
        "~/dataset/virtual_rodent/v5"))
    ap.add_argument("--phases", type=int, default=1, choices=(1, 2),
                    help="1 = segment scaling only (fast); 2 = +STAC site offsets.")
    ap.add_argument("--n-rounds", type=int, default=3)
    ap.add_argument("--m-iters", type=int, default=100)
    ap.add_argument("--ik-iters", type=int, default=500)
    ap.add_argument("--freeze", action="store_true",
                    help="freeze the 5 unmeasurable segments (scapula, neck, "
                         "pelvis, hand, foot) at rel 1.0.")
    ap.add_argument("--tag", default="",
                    help="suffix for output filenames, e.g. '_frozen'.")
    a = ap.parse_args()

    meta = os.path.join(RAMP_DIR, "dataset_meta.csv")
    trials = pick_trials(meta, a.rat, a.n_trials)
    # group trial windows by session so each big CSV is read only once
    by_session = {}
    print(f"rat={a.rat}: pooling {len(trials)} trials (window_start -> {a.end_col})")
    for idx, row in trials:
        ts = int(row["trial_start"])
        lo, hi = ts + int(row["window_start"]), ts + int(row[a.end_col])
        by_session.setdefault(row["session_name"], []).append((lo, hi))
        print(f"  row={idx} session={row['session_name']} abs[{lo}..{hi}] ({hi-lo+1}f)")

    frames = []
    for session, windows in by_session.items():
        kp_csv = os.path.join(RAMP_DIR, "per_session", session, "keypoints3d.csv")
        frames += read_windows(kp_csv, windows)
    pooled = len(frames)
    if a.max_frames and pooled > a.max_frames:
        idx = np.linspace(0, pooled - 1, a.max_frames).round().astype(int)
        frames = [frames[i] for i in idx]
    nval = [int(v.sum()) for _, v in frames]
    print(f"  POOLED {pooled} frames -> using {len(frames)} (even-spaced) "
          f"from {len(by_session)} session(s); valid-kp/frame "
          f"min={min(nval)} median={int(np.median(nval))} max={max(nval)}")

    os.makedirs(a.out_dir, exist_ok=True)
    mjb = os.path.join(a.out_dir, f"rodent_{a.rat}{a.tag}.mjb")
    label = f"{a.rat}_ramp_{len(trials)}trials{a.tag}"
    if a.phases == 1:
        fit_phase1(BASE_XML, frames, mjb, label, n_rounds=a.n_rounds,
                   m_iters=a.m_iters, ik_iters=a.ik_iters, target_weight=0.05,
                   freeze_unmeasured=a.freeze)
    else:
        fit_on_frames(BASE_XML, frames, mjb, label, n_rounds=a.n_rounds,
                      m_iters=a.m_iters, ik_iters=a.ik_iters, target_weight=0.05)

    xml = os.path.join(a.out_dir, f"rodent_{a.rat}{a.tag}.xml")
    load_make_xml().convert(a.rat, base_xml=BASE_XML, mjb=mjb, out=xml)
    fix_ankle_range(xml)
    xml_fb = os.path.join(a.out_dir, f"rodent_{a.rat}{a.tag}_nofreejoint.xml")
    convert_fixed_base(mjb, BASE_XML, xml_fb)
    fix_ankle_range(xml_fb)
    print(f"\nDONE\n  mjb: {mjb}\n  xml (free joint): {xml}\n  xml (fixed base): {xml_fb}")


if __name__ == "__main__":
    main()
