# Adjustabodies: Data-Driven Body Model Fitting

## Overview

Adjustabodies is a two-phase optimization pipeline that adapts a generic MuJoCo skeletal model to match the body proportions of individual animals, using only 3D keypoint tracking data. Starting from a base rodent model (66 bodies, 61 hinge joints, 24 tracked keypoints), the pipeline produces per-animal fitted models with 2.67--3.00 mm residual error, enabling accurate inverse kinematics and physics-based behavioral analysis.

The pipeline was developed for the Green dataset: 5 Long-Evans rats (Captain, Emilie, Heisenberg, Mario, Remy) performing a ball-fetch task, tracked at 180 fps with 24 labeled 3D keypoints.

## Base Model

The starting model is a MuJoCo rodent skeleton derived from the dm_control rodent, with data-driven joint limits:

| Property | Value |
|----------|-------|
| Bodies | 66 |
| Joints | 61 (all hinge) |
| Generalized coordinates (nq) | 61 |
| Tracked keypoints | 24 (+ 1 ball target) |
| Sites | 43 (24 keypoint sites + 19 anatomical landmarks) |
| Geoms | 88 (capsules for collision) |
| Skin | 6,880-vertex mesh (rodent_walker_skin.skn) |

The skeleton is organized into 12 scalable segments:

| Segment | Bodies | Description |
|---------|--------|-------------|
| head | 2 | Skull, jaw |
| neck | 7 | Cervical vertebrae |
| spine | 6 | Thoracic + lumbar vertebrae |
| pelvis | 1 | Pelvic girdle |
| tail | 30 | Caudal vertebrae (densely articulated) |
| scapula | 2 | Left/right scapulae |
| upper_arm | 2 | Left/right humerus |
| lower_arm | 2 | Left/right radius/ulna |
| hand | 4 | Left/right wrist + digits |
| upper_leg | 2 | Left/right femur |
| lower_leg | 2 | Left/right tibia/fibula |
| foot | 4 | Left/right ankle + digits |

## Algorithm

### Input

- **3D keypoint data**: 500 randomly sampled frames per animal (from the active behavioral phase), each frame providing 24 keypoint positions in world coordinates (mm)
- **Base MuJoCo model**: XML with skeleton topology, joint limits, and keypoint site definitions
- **Data-driven segment targets**: Median inter-keypoint distances measured from the tracking data, compared to the same distances in the base model's default pose

### Phase 0: Data-Driven Segment Measurement

Before fitting begins, we measure actual segment lengths from the 3D tracking data to establish anatomical priors:

1. **Measure inter-keypoint distances** for 16 segment pairs (e.g., shoulder-to-elbow, knee-to-ankle, 4 tail segments)
2. **Strict outlier rejection**: Per-chain filtering removes frames where any segment in a kinematic chain deviates >15% from its median length
3. **Compute target scale factors**: For each model segment, the ratio of the total measured chain length to the corresponding chain length in the base model's default pose

These targets serve as soft constraints during Phase 1, preventing the optimizer from finding degenerate solutions (e.g., inflating the tail to reduce residuals on other body parts).

### Phase 1: Segment Scaling

**Goal**: Find per-segment scale factors that minimize the distance between the model's forward-kinematics keypoint positions and the tracked 3D keypoints.

**Parameters**: 1 global scale + 12 relative segment scales = 13 parameters

**Optimization**: Alternating minimization over 6 rounds:

1. **Q-step (CPU)**: Fix segment scales, solve inverse kinematics for all 500 frames using gradient descent on the MuJoCo C API (1000 iterations per frame). This finds the joint angles that best place the keypoint sites at the tracked positions, given the current body shape.

2. **M-step (GPU)**: Fix joint angles, optimize segment scales using Adam (learning rate 0.003, 300 iterations) on differentiable forward kinematics via MuJoCo MJX. The loss function is:

$$L = L_{\text{IK}} + \lambda_{\text{reg}} \sum_i (s_i - 1)^2 + \lambda_{\text{target}} \sum_{i \in \text{measured}} (s_i^{\text{abs}} - s_i^{\text{target}})^2$$

where $s_i^{\text{abs}} = s_{\text{global}} \times s_i^{\text{rel}}$ is the absolute scale for segment $i$, and $\lambda_{\text{target}} = 0.05$ balances anatomical priors against IK fit quality. The regularization term ($\lambda_{\text{reg}} = 0.001$) penalizes deviation from unit scale, and the target term pulls measured segments toward their data-driven ratios.

**Output**: Scaled body model with modified body_pos, body_ipos, geom_pos, geom_size, site_pos, and jnt_pos arrays.

### Phase 2: STAC Site Offset Calibration

**Goal**: Fine-tune the positions where keypoint tracking labels attach to each bone, compensating for systematic offsets between the model's anatomical landmarks and the tracking system's label placement.

**Parameters**: 43 x 3 = 129 site offset parameters (with L/R symmetry enforcement reducing effective DOF to ~72)

**Key constraint**: Segment scales are completely frozen in Phase 2 (zero gradient via `optax.set_to_zero()`). Only site offsets are optimized.

**Optimization**: Same alternating structure as Phase 1, 6 rounds:

1. **Q-step (CPU)**: IK solve with the scaled body + current site offsets (1000 iterations/frame)
2. **M-step (GPU)**: Adam on site offsets only (learning rate 0.001, 300 iterations), with:
   - **L/R symmetry enforcement**: After each Adam step, left/right site pairs are averaged and mirrored. Midline sites are zeroed in the lateral (Y) axis.
   - **Offset regularization**: $\lambda_\text{offset} = 0.01$ penalizes large offsets, keeping adjustments small and physically plausible.

**Output**: Final fitted model with adjusted keypoint site positions. Typical offsets are 1--3 mm, correcting for differences between the model's joint-center-based landmarks and the tracking system's surface-based keypoint labels.

## Results

### Per-Animal Segment Scales

| Segment | Base | Captain | Emilie | Heisenberg | Mario | Remy | Average |
|---------|------|---------|--------|------------|-------|------|---------|
| head | 1.000 | 0.966 | 0.936 | 0.984 | 0.918 | 0.946 | 0.945 |
| neck | 1.000 | 0.987 | 1.023 | 1.020 | 0.930 | 1.010 | 0.994 |
| spine | 1.000 | 1.174 | 1.266 | 1.221 | 1.113 | 1.251 | 1.196 |
| pelvis | 1.000 | 0.998 | 1.035 | 1.019 | 0.932 | 1.014 | 0.998 |
| tail | 1.000 | 1.021 | 1.018 | 0.995 | 0.918 | 0.976 | 1.006 |
| scapula | 1.000 | 0.991 | 1.030 | 1.018 | 0.934 | 1.009 | 0.996 |
| upper_arm | 1.000 | 0.660 | 0.711 | 0.704 | 0.629 | 0.679 | 0.680 |
| lower_arm | 1.000 | 0.915 | 0.934 | 0.899 | 0.870 | 0.944 | 0.912 |
| hand | 1.000 | 0.971 | 0.993 | 0.962 | 0.910 | 0.975 | 0.966 |
| upper_leg | 1.000 | 1.161 | 1.235 | 1.110 | 1.093 | 1.132 | 1.158 |
| lower_leg | 1.000 | 0.792 | 0.800 | 0.832 | 0.777 | 0.791 | 0.792 |
| foot | 1.000 | 0.955 | 1.005 | 0.957 | 0.912 | 0.961 | 0.962 |

### Residual Error (mm)

| Model | After Phase 1 | After Phase 2 | Improvement |
|-------|--------------|--------------|-------------|
| Average (all 5 rats) | 3.86 | 2.98 | 22.8% |
| Captain | 3.98 | 3.00 | 24.6% |
| Emilie | 3.69 | 2.67 | 27.6% |
| Heisenberg | 4.00 | 2.85 | 28.8% |
| Mario | 3.70 | 2.80 | 24.3% |
| Remy | 3.98 | 2.84 | 28.6% |

Phase 2 (STAC site offsets) consistently reduces residuals by 23--29%, from ~3.8 mm to ~2.8 mm.

### Key Anatomical Findings

**Spine is longer than the base model** (+13--27%): All 5 rats have a longer torso than the dm_control rodent model, with Mario being the most compact (+11%) and Emilie the most elongated (+27%).

**Upper arm is shorter than the base model** (-29--37%): The base model's shoulder-to-elbow distance (65.3 mm) substantially exceeds the tracked keypoint distance (~44 mm). This likely reflects a difference between the model's glenohumeral joint center and the tracking system's surface-placed "Shoulder" keypoint.

**Tail is well-controlled** (-8% to +2%): The data-driven segment constraints successfully prevent tail inflation, which was the primary failure mode in earlier unconstrained fits (+29% in v1).

**Mario is the smallest rat**: Consistently smallest scales across nearly all segments, ~8--10% smaller than the average.

**Individual variation is significant**: Spine scale ranges from 1.11 (Mario) to 1.27 (Emilie), a 14% spread. This justifies per-animal fitting rather than using a single average model.

## Implementation

### Software Stack

- **MuJoCo 3.6**: Physics engine, C API for IK, MJX for GPU-accelerated differentiable FK
- **JAX 0.6.2**: Automatic differentiation and GPU compilation
- **Optax**: Adam optimizer with multi-transform (separate LR for scales vs offsets)
- **Python 3.10+**: Pipeline orchestration
- **LSF cluster**: Parallel per-animal fitting on NVIDIA A100 GPUs

### Computational Cost

| Stage | Time | Hardware |
|-------|------|----------|
| Data-driven segment measurement | <1s | CPU |
| Phase 1 (6 rounds, 500 frames) | ~25 min | A100 GPU + 8-core CPU |
| Phase 2 (6 rounds, 500 frames) | ~25 min | A100 GPU + 8-core CPU |
| **Total per animal** | **~50 min** | A100 GPU |
| **All 6 models (parallel)** | **~50 min** | 6x A100 GPU |

The bottleneck is the CPU IK solve (Q-step), running at ~200 fps across 8 workers. The GPU M-step takes <1s per round after initial compilation (~90s).

### Skin Grafting

The fitted .mjb models include a 6,880-vertex skin mesh for visualization. The skin is added to the base XML via MjSpec before compilation and scaling, ensuring the mesh deforms correctly with the resized skeleton. Keypoint sites (24) are preserved alongside the skin.

## Usage

```bash
# Fit a single animal
python3 scripts/fit_green_rats.py \
    --green-dir /path/to/green \
    --base-model models/rodent_data_driven_limits.xml \
    --output-dir /path/to/output \
    --only-rat captain \
    --target-weight 0.05

# Apply skin to fitted models
python3 scripts/apply_skin.py \
    --base-xml models/rodent_data_driven_limits.xml \
    --skin-xml /path/to/rodent.xml \
    --fits-dir /path/to/fitted_json \
    --output-dir /path/to/skinned_output
```

### Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--target-weight` | 0.05 | Strength of data-driven segment constraints (0 = unconstrained, 1 = rigid) |
| `--frames-per-rat` | 500 | Number of random frames sampled per animal |
| `--n-rounds` | 6 | Alternating optimization rounds per phase |
| `--m-iters` | 300 | Adam iterations per M-step |
| `--ik-iters` | 1000 | Gradient descent iterations per frame in IK |

## Design Decisions

### Why alternating optimization?

The IK problem (finding joint angles) and the scaling problem (finding body proportions) are coupled: changing the body shape changes which joint angles best explain the data. Alternating between IK and scaling is a coordinate descent approach that converges reliably, avoids the memory cost of jointly differentiating through both, and leverages the speed of MuJoCo's C API for IK.

### Why freeze scales in Phase 2?

Early versions allowed small scale adjustments in Phase 2, but the optimizer exploited this by inflating the global scale ~10% to reduce site offset magnitudes. Freezing scales ensures Phase 2 only adjusts keypoint attachment points, maintaining the anatomically-grounded proportions from Phase 1.

### Why soft constraints rather than hard constraints on segment lengths?

Hard constraints (weight=1.0) pin segments exactly to the measured ratios, leaving no room for the IK solver to find a better fit. The data-driven measurements have their own noise (tracking errors, keypoint placement ambiguity), so a soft constraint (weight=0.05) provides a good prior while allowing the optimizer to deviate when the IK evidence is strong.
