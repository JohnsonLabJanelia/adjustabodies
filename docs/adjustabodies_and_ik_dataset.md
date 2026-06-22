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

- **Phase 2 — STAC site calibration** (`stac.py`): **129 site-offset parameters**
  (43 sites × 3, reduced to ~72 effective DOF by enforced L/R symmetry). Adjusts
  *where* each tracking label attaches to its bone. **Segment scales are frozen
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
