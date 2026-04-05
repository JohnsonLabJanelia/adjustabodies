#!/usr/bin/env python3
"""GPU UMAP embedding of pose features using RAPIDS cuML.

Reads preprocessed numpy arrays (from preprocess_ik_for_umap.py),
runs z-score → PCA → UMAP on GPU for three feature sets:
  1. qpos (hinge joint angles) — posture
  2. qvel (hinge joint velocities) — movement
  3. qpos+qvel (combined) — behavioral state

All 4.3M frames at 180fps are embedded to preserve stride cycle
trajectories through the UMAP space.

Outputs (in --output-dir, same as input):
  umap_qpos.npy      [N, 2] float32
  umap_qvel.npy      [N, 2] float32
  umap_combined.npy   [N, 2] float32
  umap_params.json    PCA variance explained, UMAP params, timing

Requires: conda env with cuml (RAPIDS), cupy
    conda install -c rapidsai -c conda-forge cuml=26.02 cuda-version=12.6

Usage:
    python3 scripts/gpu_umap.py \
        --data-dir /path/to/umap_data \
        --n-pca-pos 20 --n-pca-vel 20 --n-pca-combined 30 \
        --n-neighbors 15 --min-dist 0.1
"""

import argparse
import json
import os
import time
import numpy as np


def gpu_embed(data_np, name, n_pca, n_components, n_neighbors, min_dist, seed):
    """z-score → PCA → UMAP on GPU. Returns (embedding, info_dict)."""
    import cupy as cp
    from cuml.decomposition import PCA
    from cuml.manifold.umap import UMAP

    info = {'name': name, 'input_shape': list(data_np.shape)}
    t0 = time.time()

    # Transfer to GPU
    data = cp.asarray(data_np, dtype=cp.float32)

    # Z-score normalize, drop zero-variance columns (e.g. unused joints)
    mean = cp.mean(data, axis=0)
    std = cp.std(data, axis=0)
    nonzero_mask = std > 1e-8
    n_dropped = int((~nonzero_mask).sum())
    if n_dropped > 0:
        print(f"  Dropping {n_dropped} zero-variance columns "
              f"({int(nonzero_mask.sum())} remaining)")
        data = data[:, nonzero_mask]
        std = std[nonzero_mask]
        mean = mean[nonzero_mask]
    data = (data - mean) / std
    info['n_dropped_cols'] = n_dropped
    info['t_zscore'] = time.time() - t0

    # PCA (cap components to available dimensions)
    t1 = time.time()
    n_pca = min(n_pca, data.shape[1] - 1)
    pca = PCA(n_components=n_pca)
    data_pca = pca.fit_transform(data)
    var_explained = float(cp.asnumpy(pca.explained_variance_ratio_).sum())
    info['n_pca'] = n_pca
    info['variance_explained'] = var_explained
    info['t_pca'] = time.time() - t1
    print(f"  PCA({n_pca}): {var_explained:.1%} variance, {info['t_pca']:.1f}s")

    # Free original data from GPU
    del data
    cp.get_default_memory_pool().free_all_blocks()

    # UMAP
    t2 = time.time()
    umap = UMAP(
        n_neighbors=n_neighbors,
        n_components=n_components,
        min_dist=min_dist,
        metric='euclidean',
        random_state=seed,
    )
    embedding_gpu = umap.fit_transform(data_pca)
    info['t_umap'] = time.time() - t2
    print(f"  UMAP(nn={n_neighbors}, md={min_dist}): {info['t_umap']:.1f}s")

    embedding = cp.asnumpy(embedding_gpu).astype(np.float32)
    info['t_total'] = time.time() - t0
    info['output_shape'] = list(embedding.shape)

    del data_pca, embedding_gpu
    cp.get_default_memory_pool().free_all_blocks()

    return embedding, info


def main():
    parser = argparse.ArgumentParser(description="GPU UMAP embedding of pose features")
    parser.add_argument('--data-dir', required=True, help="Directory with .npy files")
    parser.add_argument('--n-pca-pos', type=int, default=20, help="PCA components for qpos")
    parser.add_argument('--n-pca-vel', type=int, default=20, help="PCA components for qvel")
    parser.add_argument('--n-pca-combined', type=int, default=30, help="PCA components for combined")
    parser.add_argument('--n-components', type=int, default=3, help="UMAP output dims (2 or 3)")
    parser.add_argument('--n-neighbors', type=int, default=15)
    parser.add_argument('--min-dist', type=float, default=0.1)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--skip-existing', action='store_true',
                        help="Skip embeddings that already exist")
    args = parser.parse_args()

    # Check GPU
    import cupy as cp
    print(f"GPU: {cp.cuda.runtime.getDeviceProperties(0)['name'].decode()}")
    print(f"VRAM: {cp.cuda.runtime.memGetInfo()[1] / 1e9:.1f} GB")

    # Load data
    print(f"\nLoading data from {args.data_dir}/")
    qpos = np.load(os.path.join(args.data_dir, 'qpos_hinges.npy'))
    qvel = np.load(os.path.join(args.data_dir, 'qvel_hinges.npy'))
    print(f"  qpos: {qpos.shape} ({qpos.nbytes/1e6:.0f} MB)")
    print(f"  qvel: {qvel.shape} ({qvel.nbytes/1e6:.0f} MB)")

    N = qpos.shape[0]
    combined = np.concatenate([qpos, qvel], axis=1)
    print(f"  combined: {combined.shape} ({combined.nbytes/1e6:.0f} MB)")

    configs = [
        ('qpos', qpos, args.n_pca_pos),
        ('qvel', qvel, args.n_pca_vel),
        ('combined', combined, args.n_pca_combined),
    ]

    all_info = {}

    for name, data, n_pca in configs:
        out_path = os.path.join(args.data_dir, f'umap_{name}.npy')

        if args.skip_existing and os.path.exists(out_path):
            print(f"\n=== {name}: SKIPPED (exists) ===")
            continue

        print(f"\n{'='*60}")
        print(f"Embedding: {name} ({data.shape[0]} × {data.shape[1]} → PCA {n_pca} → UMAP {args.n_components}D)")
        print(f"{'='*60}")

        embedding, info = gpu_embed(
            data, name, n_pca, args.n_components,
            args.n_neighbors, args.min_dist, args.seed)

        np.save(out_path, embedding)
        print(f"  Saved: {out_path} ({embedding.nbytes/1e6:.0f} MB)")
        print(f"  Total: {info['t_total']:.1f}s")

        all_info[name] = info

    # Save parameters
    params_path = os.path.join(args.data_dir, 'umap_params.json')
    params = {
        'n_frames': int(N),
        'n_neighbors': args.n_neighbors,
        'min_dist': args.min_dist,
        'seed': args.seed,
        'embeddings': all_info,
    }
    with open(params_path, 'w') as f:
        json.dump(params, f, indent=2)
    print(f"\nParams: {params_path}")

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    for name, info in all_info.items():
        print(f"  {name:12s}: PCA {info['variance_explained']:.1%} var, "
              f"UMAP {info['t_umap']:.0f}s, total {info['t_total']:.0f}s")
    print()


if __name__ == "__main__":
    main()
