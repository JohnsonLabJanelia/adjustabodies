#!/bin/bash
# cluster_batch_ik_green.sh — Run MJX batch IK on the full Green dataset
#
# Processes ~4100 trials (4.3M frames) from green_traj3d.bin on A100 GPU.
# Uses adjustabodies.ik_mjx with scan mode (full JIT) for maximum throughput.
#
# Usage:
#   bash scripts/cluster_batch_ik_green.sh [--dry-run]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"

GREEN_DIR="/groups/johnson/johnsonlab/virtual_rodent/green"
MODEL="/groups/johnson/johnsonlab/virtual_rodent/body_model/rodent_red_mj_dev_fitted.mjb"
OUTPUT="$GREEN_DIR/qpos_export_mjx.csv"
CONDA_ENV="mjx"
MAX_ITERS=1000
LR=0.01
BATCH_SIZE=512
DRY_RUN=false
RUN_MODE=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --dry-run)      DRY_RUN=true; shift ;;
        --run)          RUN_MODE=true; shift ;;
        --model)        MODEL="$2"; shift 2 ;;
        --output)       OUTPUT="$2"; shift 2 ;;
        --max-iters)    MAX_ITERS="$2"; shift 2 ;;
        --lr)           LR="$2"; shift 2 ;;
        --batch-size)   BATCH_SIZE="$2"; shift 2 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

if $RUN_MODE; then
    echo "=== Green Batch IK (MJX GPU) ==="
    echo "Host: $(hostname)"
    echo "Date: $(date)"
    echo "Data: $GREEN_DIR"
    echo "Model: $MODEL"
    echo "Output: $OUTPUT"
    echo "Params: iters=$MAX_ITERS lr=$LR batch=$BATCH_SIZE"

    source ~/miniconda3/bin/activate
    conda activate $CONDA_ENV

    python3 -c "
import jax
print(f'JAX {jax.__version__}, devices: {jax.devices()}')
gpu = [d for d in jax.devices() if d.platform == 'gpu']
assert len(gpu) > 0, 'No GPU found!'
print(f'GPU: {gpu[0]}')
"

    pip install -e "$REPO_DIR" --quiet 2>&1 | tail -3

    # Run the batch IK script
    python3 << PYEOF
import numpy as np
import time
import sys
sys.path.insert(0, "$REPO_DIR")

from adjustabodies.model import load_model, build_site_indices
from adjustabodies.species.rodent import RAT24_SITES
from adjustabodies.ik_mjx import build_ik_solver, IKConfig

# ── Load model ──────────────────────────────────────────────
m = load_model("$MODEL", add_free_joint=True, fix_geoms_for_mjx=True)
site_ids = build_site_indices(m, RAT24_SITES)
print(f"Model: nq={m.nq} nv={m.nv} sites={sum(1 for s in site_ids if s>=0)}/24")

# ── Load Green trajectory data ──────────────────────────────
# Green uses a custom binary format, not RED CSV
import struct

MAGIC = 0x024E5247
traj_path = "$GREEN_DIR/repaired_traj3d.bin"
print(f"Loading: {traj_path}")

data = np.memmap(traj_path, dtype=np.uint8, mode='r')
magic = struct.unpack_from('<I', data, 0)[0]
assert magic == MAGIC, f"Bad magic: {magic:#x}"
num_trials = struct.unpack_from('<I', data, 8)[0]
num_fields = struct.unpack_from('<I', data, 12)[0]
fps = struct.unpack_from('<I', data, 20)[0]
num_kp = struct.unpack_from('<H', data, 24)[0]
print(f"Trials: {num_trials}, kp: {num_kp}, fps: {fps}")

# Parse fields
FIELD_DESC_SIZE = 44
FIELD_NAME_LEN = 32
fields = []
cumulative = 0
for i in range(num_fields):
    pos = 32 + i * FIELD_DESC_SIZE
    name = bytes(data[pos:pos+FIELD_NAME_LEN]).split(b'\0')[0].decode()
    epf = struct.unpack_from('<I', data, pos + 32)[0]
    esz = struct.unpack_from('<I', data, pos + 36)[0]
    fields.append({'name': name, 'epf': epf, 'esz': esz, 'offset': cumulative})
    cumulative += epf * esz

traj3d_field = next(f for f in fields if f['name'] == 'traj3d')
bytes_per_frame = cumulative

# Parse index
desc_end = 32 + num_fields * FIELD_DESC_SIZE
index_start = (desc_end + 7) & ~7
INDEX_ENTRY_SIZE = 12
trials_index = []
for i in range(num_trials):
    pos = index_start + i * INDEX_ENTRY_SIZE
    offset = struct.unpack_from('<Q', data, pos)[0]
    nf = struct.unpack_from('<I', data, pos + 8)[0]
    trials_index.append((offset, nf))

total_frames = sum(nf for _, nf in trials_index)
print(f"Total frames: {total_frames}")

# ── Build solver ────────────────────────────────────────────
import jax
import jax.numpy as jnp

config = IKConfig(max_iters=$MAX_ITERS, lr=$LR, batch_size=$BATCH_SIZE, use_scan=True)
print(f"Building solver (scan mode, GPU)...")
t0 = time.time()
solve_batch, nq = build_ik_solver(m, site_ids, config)

# Warmup compilation
dummy_kp = jnp.zeros((config.batch_size, 24, 3))
dummy_v = jnp.ones((config.batch_size, 24))
_ = solve_batch(dummy_kp, dummy_v)
jax.block_until_ready(_[0])
print(f"Compiled in {time.time()-t0:.1f}s")

# ── Process all trials ──────────────────────────────────────
scale = 0.001  # mm → m
stride = traj3d_field['epf']
traj_offset = traj3d_field['offset']

output_path = "$OUTPUT"
print(f"Output: {output_path}")

with open(output_path, 'w') as fout:
    fout.write(f"# GREEN mjx_batch_ik export (adjustabodies)\n")
    fout.write(f"# model: $MODEL\n")
    fout.write(f"# nq: {nq}\n")
    fout.write(f"# max_iters: $MAX_ITERS\n")
    fout.write(f"# lr: $LR\n")
    cols = ["trial", "frame"] + [f"qpos_{i}" for i in range(nq)] + ["residual_mm"]
    fout.write(",".join(cols) + "\n")

    t_solve = time.time()
    frames_done = 0

    for trial_idx in range(num_trials):
        trial_offset, nf = trials_index[trial_idx]

        # Extract keypoints for this trial
        start = trial_offset + nf * traj_offset
        nbytes = nf * stride * 4
        traj = np.frombuffer(data[start:start+nbytes], dtype=np.float32).reshape(nf, stride)

        # First 72 floats = 24 kp × 3 coords (skip ball at index 24)
        kp3d = traj[:, :72].reshape(nf, 24, 3).copy() * scale
        valid = np.isfinite(kp3d).all(axis=-1) & ~(kp3d == 0).all(axis=-1)
        valid = valid.astype(np.float32)

        # Process in batches
        for batch_start in range(0, nf, config.batch_size):
            batch_end = min(batch_start + config.batch_size, nf)
            actual = batch_end - batch_start

            kp_batch = kp3d[batch_start:batch_end]
            v_batch = valid[batch_start:batch_end]

            if actual < config.batch_size:
                kp_batch = np.concatenate([kp_batch, np.zeros((config.batch_size - actual, 24, 3), dtype=np.float32)])
                v_batch = np.concatenate([v_batch, np.zeros((config.batch_size - actual, 24), dtype=np.float32)])

            qpos_b, res_b = solve_batch(jnp.array(kp_batch), jnp.array(v_batch))
            qpos_np = np.array(qpos_b[:actual])
            res_np = np.array(res_b[:actual]) * 1000.0

            for f in range(actual):
                frame_idx = batch_start + f
                parts = [str(trial_idx), str(frame_idx)]
                parts += [f"{q:.8f}" for q in qpos_np[f]]
                parts += [f"{res_np[f]:.4f}"]
                fout.write(",".join(parts) + "\n")

            frames_done += actual

        if trial_idx % 100 == 0 or trial_idx == num_trials - 1:
            elapsed = time.time() - t_solve
            fps_rate = frames_done / max(elapsed, 0.001)
            print(f"  Trial {trial_idx}/{num_trials}  "
                  f"{frames_done}/{total_frames} frames  "
                  f"{fps_rate:.0f} fps  {elapsed:.0f}s")

    elapsed = time.time() - t_solve
    print(f"\nDone: {frames_done} frames in {elapsed:.1f}s "
          f"({frames_done/max(elapsed,0.001):.0f} fps)")
    print(f"Output: {output_path}")
PYEOF

    echo ""
    echo "=== Batch IK Complete ==="
    exit 0
fi

# Submit mode
LOG="$GREEN_DIR/batch_ik_mjx.log"

BSUB_CMD="bsub -W 4:00 -n 8 -gpu \"num=1\" -q gpu_a100 -P johnson \
    -J green_batch_ik \
    -o $LOG \
    bash $REPO_DIR/scripts/cluster_batch_ik_green.sh --run"

echo "Green Batch IK (MJX GPU)"
echo "  Data:   $GREEN_DIR"
echo "  Model:  $MODEL"
echo "  Output: $OUTPUT"
echo "  Params: iters=$MAX_ITERS lr=$LR batch=$BATCH_SIZE"
echo "  Log:    $LOG"
echo ""

if $DRY_RUN; then
    echo "[dry-run] $BSUB_CMD"
else
    eval "$BSUB_CMD"
    echo ""
    echo "Monitor: bjobs -w | grep green_batch"
    echo "Log:     tail -f $LOG"
fi
