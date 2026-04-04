# adjustabodies

MuJoCo body model fitting for behavioral neuroscience. Resize body segments, calibrate site positions (STAC), and run batch IK for multi-camera 3D pose estimation.

Supports rodent (rat/mouse) and fly body models from the [janelia-anibody](https://github.com/janelia-anibody) project.

## Features

- **Body segment scaling**: Per-segment resize with L/R symmetry (gradient-based via MJX on GPU)
- **STAC site calibration**: Site offset optimization with bilateral symmetry enforcement
- **Batch IK**: CPU and GPU (MJX) inverse kinematics solvers
- **Arena transform**: Calibration frame ↔ MuJoCo frame coordinate conversion
- **Model I/O**: Load/save fitted models as MJB with JSON metadata

## Installation

```bash
# CPU only (model loading, IK, data I/O)
pip install git+https://github.com/JohnsonLabJanelia/adjustabodies.git

# With GPU support (MJX fitting, batch IK on cluster)
pip install "adjustabodies[mjx] @ git+https://github.com/JohnsonLabJanelia/adjustabodies.git"
```

## Quick start

```python
from adjustabodies import model, io, ik_cpu
from adjustabodies.species.rodent import RODENT_CONFIG, RAT24_SITES
from adjustabodies.arena import ArenaTransform

# Load model
m = model.load_model("models/rodent_data_driven_limits.xml")
site_ids = model.build_site_indices(m)

# Load keypoints with arena transform
arena_tf = ArenaTransform.from_session("project/mujoco_session.json")
frames = io.load_keypoints3d("project/labeled_data/.../keypoints3d.csv", arena_tf=arena_tf)

# Batch IK
all_qpos = ik_cpu.batch_ik_cpu(m, frames, site_ids, max_iters=1000)
```

## Used by

- **[RED](https://github.com/JohnsonLabJanelia/red)** — GPU-accelerated 3D multi-camera keypoint labeling
- **[JARVIS-HybridNet](https://github.com/JARVIS-MoCap/JARVIS-HybridNet)** — Multi-view 3D pose estimation
