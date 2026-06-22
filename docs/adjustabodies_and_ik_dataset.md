# Adjustabodies and the Per-Rat Inverse-Kinematics Dataset

*Johnson Lab, Janelia — short technical overview*

## Summary

**Adjustabodies** is an internal tool that adapts a single generic MuJoCo rodent
skeleton into five *individualized* body models — one per rat in the Green
ball-fetch cohort (Captain, Emilie, Heisenberg, Mario, Remy) — using nothing but
3D keypoint tracking data. Those fitted models are then used to run batch
**inverse kinematics (IK)**, converting every tracked frame into a full vector of
MuJoCo joint angles. The result is the **qpos_v4 dataset**: ~2.54 million frames
of 68-DOF per-rat pose, the joint-angle substrate for downstream gait and
locomotion analysis.

Two products, one pipeline:

1. **Five fitted body models** (`.mjb`) capturing each rat's individual proportions.
2. **The qpos_v4 IK dataset** — per-frame generalized coordinates for all 5 rats.

---

## Part 1 — Adjustabodies (the body-model fitter)

### What problem it solves

A single off-the-shelf rodent model does not match any real rat. Limb and torso
proportions vary enough between animals (spine length varies 14% across our five
rats) that forcing one model onto all of them injects systematic error into any
IK solve. Adjustabodies fits the *shape* of the model to each animal first, so
the later joint-angle solve is anatomically honest.

It needs only data we already have — 3D keypoints — and no per-animal CT scans,
manual rigging, or markers.

### Base model

Derived from the dm_control rodent, with data-driven joint limits
(`models/rodent_data_driven_limits.xml`):

| Property | Value |
|---|---|
| Bodies | 66 |
| Hinge joints | 61 |
| Tracked keypoints | 24 (+1 ball target) |
| Sites | 43 (24 keypoint + 19 anatomical) |
| Geoms | 88 collision capsules |
| Skin | 6,880-vertex mesh (visualization) |

The skeleton is grouped into **12 bilaterally-symmetric scalable segments**: head,
neck, spine, pelvis, tail, scapula, upper_arm, lower_arm, hand, upper_leg,
lower_leg, foot. The canonical definitions (site names, segment groups, L/R
symmetry pairs, solver config) live in `adjustabodies/species/rodent.py` and are
the single source of truth that the lab's C++ tooling (RED) mirrors.

### Fitting procedure — two phases of coordinate descent

Each per-rat fit consumes **500 frames** of 24-keypoint 3D data (transformed from
arena mm into MuJoCo meters) and runs an alternating optimization: a CPU IK
"Q-step" (find joint angles given current shape) alternates with a GPU MJX
"M-step" (find shape parameters given current angles), 6 rounds per phase.

- **Phase 0 — segment measurement** (`segment_lengths.py`): median inter-keypoint
  distances over 16 segment pairs, with strict per-chain outlier rejection (drop
  any frame deviating >15% from the segment median). These become *soft* anatomical
  targets — chiefly to stop the optimizer inflating the tail to absorb error.

- **Phase 1 — segment scaling** (`resize.py`): **13 parameters** (1 global + 12
  relative segment scales), Adam through differentiable forward kinematics. Loss =
  keypoint FK error + unit-scale regularization + a soft pull toward the Phase-0
  measured ratios.

- **Phase 2 — STAC site calibration** (`stac.py`): adjusts *where* each tracking
  keypoint attaches to its bone. The offset array nominally spans all 43 model
  sites, but only the **24 tracked keypoint sites** enter the loss — the 19
  non-keypoint anatomical landmarks have no keypoint target and are held at ~0 by
  regularization. So the calibrated free parameters are **24 keypoints × 3 = 72
  offsets**, further reduced by enforced L/R symmetry. **Segment scales are frozen
  here** (`optax.set_to_zero()`) — an early version let the optimizer cheat by
  inflating global scale ~10% to shrink offset magnitudes.

### Results

| Model | Residual after Phase 1 | After Phase 2 |
|---|---|---|
| Average (all 5) | 3.86 mm | **2.98 mm** |
| Captain | 3.98 | 3.00 |
| Emilie | 3.69 | **2.67** |
| Heisenberg | 4.00 | 2.85 |
| Mario | 3.70 | 2.80 |
| Remy | 3.98 | 2.84 |

Phase 2 consistently cuts residual error 23–29%, to **~2.8 mm**.

Anatomical findings that justify per-animal fitting:
- **Spine longer than base** in all 5 rats (+11% Mario … +27% Emilie).
- **Upper arm shorter** (−29 to −37%): the model's glenohumeral joint center sits
  deeper than the tracker's surface-placed "Shoulder" label.
- **Tail well-controlled** (−8% to +2%), versus +29% inflation in unconstrained v1.
- **Mario** is the smallest rat across nearly every segment.

### Cost & stack

MuJoCo 3.6 (C-API IK + MJX differentiable FK), JAX 0.6.2, Optax. ~50 min/animal on
an A100; all run in parallel on the LSF cluster. CLI entry points:
`adjustabodies-fit`, `adjustabodies-ik`, `adjustabodies-ik-mjx`.

---

## Part 2 — The qpos_v4 IK dataset

### What it is

Using the five fitted models, batch IK was run over the full tracked recordings to
produce per-frame **generalized coordinates (qpos)** — the joint angles that place
the model's 24 keypoint sites onto the tracked 3D keypoints, frame by frame.

> **DOF count.** The base *skeleton* has 61 hinge joints (nq = 61). The IK-export
> models add a 7-DOF free root joint, so the exported pose vector is **nq = 68**
> (`qpos_0 … qpos_67`). The two phases of the fitter report the skeleton's 61;
> the dataset on disk carries the full 68.

qpos layout (per `gait_ik/01_stage_outbound_qpos.py`):
- `q0…q6` — free root: `px py pz qw qx qy qz`
- `q7…q67` — 61 hinge joints (spine, hindlimbs, tail, neck/head, forelimbs)
- Limb index groups: **FootL** q13–17, **FootR** q18–22, **HandL** q56–61, **HandR** q62–67

Full per-column joint names (from `QPOS_NAMES`, `gait_ik/01_stage_outbound_qpos.py`),
grouped by body region:

| Region | Idx | Joint names |
|---|---|---|
| **Root** (free joint) | q0–q6 | `root_px`, `root_py`, `root_pz`, `root_qw`, `root_qx`, `root_qy`, `root_qz` |
| **Spine** (thoracic/lumbar) | q7–q12 | `vertebra_1_extend`, `vertebra_2_bend`, `vertebra_3_twist`, `vertebra_4_extend`, `vertebra_5_bend`, `vertebra_6_twist` |
| **Hindlimb L** (FootL) | q13–q17 | `hip_L_supinate`, `hip_L_abduct`, `hip_L_extend`, `knee_L`, `ankle_L` |
| **Hindlimb R** (FootR) | q18–q22 | `hip_R_supinate`, `hip_R_abduct`, `hip_R_extend`, `knee_R`, `ankle_R` |
| **Tail** (caudal) | q23–q46 | `vertebra_C1_extend`, `vertebra_C1_bend`, `vertebra_C2_extend`, `vertebra_C2_bend`, `vertebra_C3_extend`, `vertebra_C3_bend`, `vertebra_C4_extend`, `vertebra_C4_bend`, `vertebra_C5_extend`, `vertebra_C5_bend`, `vertebra_C6_extend`, `vertebra_C6_bend`, `vertebra_C7_extend`, `vertebra_C9_bend`, `vertebra_C11_extend`, `vertebra_C13_bend`, `vertebra_C15_extend`, `vertebra_C17_bend`, `vertebra_C19_extend`, `vertebra_C21_bend`, `vertebra_C23_extend`, `vertebra_C25_bend`, `vertebra_C27_extend`, `vertebra_C29_bend` |
| **Neck / head** | q47–q55 | `vertebra_cervical_5_extend`, `vertebra_cervical_4_bend`, `vertebra_cervical_3_twist`, `vertebra_cervical_2_extend`, `vertebra_cervical_1_bend`, `vertebra_axis_twist`, `vertebra_atlant_extend`, `atlas`, `mandible` |
| **Forelimb L** (HandL) | q56–q61 | `scapula_L_supinate`, `scapula_L_abduct`, `scapula_L_extend`, `shoulder_L`, `shoulder_sup_L`, `elbow_L` |
| **Forelimb R** (HandR) | q62–q67 | `scapula_R_supinate`, `scapula_R_abduct`, `scapula_R_extend`, `shoulder_R`, `shoulder_sup_R`, `elbow_R` |

(7 root + 6 spine + 5+5 hind + 24 tail + 9 neck/head + 6+6 fore = **68**. The same
`QPOS_NAMES` ordering applies to all 5 per-rat models — only the body proportions
differ, not the joint topology.)

### Location, files, scale

`/Volumes/johnsonlab/virtual_rodent/green/qpos_v4/qpos_<rat>.csv`
(PRFS mount; read locally, **never** stream from PRFS in a hot loop).

| File | Size | Frames |
|---|---|---|
| qpos_captain.csv | 418 MB | 524,714 |
| qpos_emilie.csv | 398 MB | 500,458 |
| qpos_heisenberg.csv | 428 MB | 538,173 |
| qpos_mario.csv | 451 MB | 567,569 |
| qpos_remy.csv | 323 MB | 406,093 |
| **Total** | **~2.0 GB** | **2,537,007** |

CSV layout: 5 `#` comment lines (model path, `nq: 68`, warm/cold iter counts),
then header `trial,frame,qpos_0,…,qpos_67,residual_mm`. Per-frame IK residual is
~7–9 mm median. (A merged single-file `qpos_v4.csv` with an added `animal` column
also exists alongside the per-rat directory.)

### How it was generated

- Solver: `adjustabodies.ik_cpu.batch_ik_cpu_trial`, warm-started across consecutive
  frames (frame 0 cold = 1000 iters, rest warm = 200 iters, lr 0.01), driver
  `scripts/batch_ik_warmstart.py`.
- Models: per-rat `rodent_green_<rat>.mjb` from `green_fits_v4/`.
- Input: 3D keypoint trajectories trimmed to per-trial analysis windows.
- Run per-animal on LSF (~4,000–4,800 s/rat); a single combined job exceeded the
  runtime limit, hence the per-rat split.

### Changes from the baseline RED IK solver

The CPU solver is a Python port of **RED's `mujoco_ik.h`** (momentum gradient descent
on site Jacobians) — the algorithm the lab used before. Three changes were needed to
get from that baseline to a clean whole-dataset solve. Joint *limits* themselves were
also changed once, at the model level.

**0. Data-driven joint limits (model, not solver).** The base model
`rodent_data_driven_limits.xml` replaces the dm_control rodent's default joint ranges
with empirical limits — **99th percentile of each joint's observed excursion + 20%
margin**. *Why:* the default ranges are loose enough to admit anatomically impossible
poses that still fit the keypoints, so the IK can wander into implausible local minima.
The data-driven ranges (e.g. spine joints at ±0.5236 rad) keep the solve in the region
the animals actually occupy. These limits are **global** — shared by all 5 per-rat
models. Per-rat fitting scales `jnt_pos` (joint anchor positions) but never `jnt_range`,
so no joint's limits were retuned per-animal or for the full-dataset pass.

**1. Joint-limit clamping restored in the CPU solver** (`ik_cpu.py:60-67`, commit
`59f40ac`). The Python port had dropped the post-integration clamp present in the C++
(`mujoco_ik.h` lines 243-249). *Why it mattered:* without it, joints integrated straight
past their limits — the commit records the spine reaching **−6.9 rad against a ±0.5
limit** — producing garbage qpos and a diffuse, uninterpretable UMAP. The fix clamps
every limited hinge/slide joint to its range after each `mj_integratePos` step. Note the
data-driven limits of change 0 only actually *bind* because of this clamp; the two go
together.

**2. MJX velocity-space gradient descent** (`ik_mjx.py`, commit `97b450f`). The GPU
solver's first implementation did gradient descent directly on the 4 raw quaternion
components of the free joint, which is wrong on the rotation manifold (the quaternion
drifts off the unit sphere and the renormalization corrupts the gradient) and gave
**27 mm residuals**. *Fix:* take gradients with respect to velocity-space (nv-dim)
perturbations that flow through a JAX exponential-map implementation of
`mj_integratePos`, so updates stay on the manifold. Validated against the CPU solver at
cosine similarity 0.9998 and <2 mm residual difference.

**3. Warm-starting for the whole-dataset run** (`batch_ik_cpu_trial`, commits `cbbbf08`
/ `8cbf6db`). This is the only change specific to scaling up from fitting (500 sampled
frames) to the 2.5M-frame export. During fitting every frame is solved cold (1000
iters). For the export, only frame 0 of each trial is cold (1000 iters); each subsequent
frame warm-starts from the previous frame's solution with **200 iters, lr 0.01, and
momentum carried across frames** (`batch_ik_warmstart.py:109`). *Why:* consecutive
180 fps frames are nearly identical, so a warm start converges in a fraction of the
iterations and the carried state yields temporally smoother trajectories — without it
the per-animal jobs exceeded the cluster runtime limit.

#### Room for further improvement

- **Residuals are still ~7–9 mm median** in the export (vs ~2.8 mm on the 500-frame fits
  used for model fitting). The gap is partly the warm-start iteration budget (200 vs
  1000) and partly that the fit frames were quality-filtered while the export solves every
  frame, including poorly-tracked ones. An adaptive iteration count (more iters when the
  per-frame residual stays high) would recover accuracy on the hard frames without paying
  for it everywhere.
- **No temporal smoothness term.** Each frame is solved independently (warm-start only
  seeds the optimizer; it doesn't penalize frame-to-frame jerk). Adding a velocity/
  acceleration regularizer — or a light forward–backward pass — would suppress residual
  per-frame jitter that currently has to be band-pass filtered out downstream
  (`02_limb_phase_pca.py`).
- **Hard joint clamping is non-smooth.** Clamping projects onto the limit box, which can
  stall a joint exactly at its bound and zero its gradient. A soft limit penalty
  (`solreflimit`-style spring, already defined in the XML defaults) inside the loss would
  let joints approach limits smoothly instead of sticking.
- **The 24-keypoint set under-constrains some DOF.** The tail carries **24 DOF but only
  5 keypoints** (`tailbase` + `tail1Q/tailmid/tail3Q/tailtip`), so the inter-vertebral
  angles are largely interpolated. And while each forelimb *has* a `shoulder` keypoint,
  the axial-rotation DOF (`shoulder_sup_*`) is geometrically near-unobservable from a
  single point on the limb, so those columns are weakly identified. More tracked points,
  or a stronger pose prior on those chains, would make them trustworthy.
- **GPU solver is validated but unused for the export.** The export ran on CPU; the MJX
  solver (change 2) reproduces it to <2 mm and could cut wall-clock substantially if the
  per-frame warm-start dependency were restructured into batched blocks.

### How it is consumed (yellow `scripts/gait_ik/`)

The raw PRFS CSVs are touched **only** by staging scripts, which write compact local
`.npz` artifacts that everything downstream reads instead:

- `01_stage_outbound_qpos.py` — outbound-phase staging. Frame join convention:
  **`v5_frame = qpos.frame − window_start`** (window_start from the v5 modeling
  export). Filters to ramp_curr Complete trials → 1,922 trials, 385,202 outbound rows.
- `01b_stage_fulltrial_qpos.py` — full-trial staging (pre/out/catch/in phases),
  reading only the four limbs' columns → 946,376 labeled rows.
- `02_limb_phase_pca.py` — per-limb limit-cycle phase: shared symmetrized PCA per
  limb pair (forelimb 6-DOF, hindlimb 5-DOF), phase = Hilbert of bandpassed PC1.

Canonical downstream table: `scripts/gait_ik/output/per_frame_gait_fulltrial.csv.gz`
(915,442 frames). The full 2.54M-frame label set
(`green_gait_labels.csv.gz` / `.bin`) is handed back to green for UMAP coloring,
joined on `(animal, trial_id, frame)`.

> **Two unrelated "v" counters.** Yellow **v5** = the modeling export (24 keypoints +
> COM scalars). Green IK **v4** = this 68-DOF qpos solve. The version numbers are
> independent.

---

## Key files for reference

| Topic | Path |
|---|---|
| Algorithm description (authoritative) | `adjustabodies/docs/adjustabodies_algorithm.md` |
| Canonical body-model definitions | `adjustabodies/adjustabodies/species/rodent.py` |
| Fit orchestration | `adjustabodies/adjustabodies/fit.py` |
| Phase 1 / Phase 2 | `adjustabodies/adjustabodies/resize.py`, `stac.py` |
| IK solvers (CPU / GPU) | `adjustabodies/adjustabodies/ik_cpu.py`, `ik_mjx.py` |
| Batch IK driver | `adjustabodies/scripts/batch_ik_warmstart.py` |
| Dataset staging (yellow) | `yellow/scripts/gait_ik/01_stage_outbound_qpos.py`, `01b_…`, `02_…` |
