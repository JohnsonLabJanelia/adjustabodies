#!/usr/bin/env python3
"""GPU UMAP embedding of egocentric 3D keypoints using RAPIDS cuML.

Reads preprocessed egocentric features (from preprocess_ego_umap.py),
runs z-score → PCA → UMAP on GPU.

Outputs (in same directory as input):
  umap_ego.npy        [N, 3] float32  — 3D UMAP embedding
  umap_params.json    PCA variance, UMAP params, timing

Requires: conda env with cuml (RAPIDS), cupy
    conda install -c rapidsai -c conda-forge cuml=26.02 cuda-version=12.6

Usage:
    python3 scripts/gpu_umap_ego.py \
        --data-dir /path/to/umap_ego \
        --n-pca 20 --n-neighbors 30 --min-dist 0.05
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

    data = cp.asarray(data_np, dtype=cp.float32)

    # Z-score normalize
    mean = cp.mean(data, axis=0)
    std = cp.std(data, axis=0)
    nonzero = std > 1e-6
    n_dropped = int((~nonzero).sum())
    n_remaining = int(nonzero.sum())
    if n_dropped > 0:
        print(f"  Dropping {n_dropped} zero-variance columns ({n_remaining} remaining)")
    data = data[:, nonzero]
    data = (data - mean[nonzero]) / std[nonzero]
    data = cp.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)

    info['n_dropped_cols'] = n_dropped
    info['t_zscore'] = time.time() - t0

    # PCA
    t1 = time.time()
    n_pca = max(2, min(n_pca, n_remaining - 1))
    pca = PCA(n_components=n_pca)
    data_pca = pca.fit_transform(data)
    var_explained = float(cp.asnumpy(pca.explained_variance_ratio_).sum())
    info['n_pca'] = n_pca
    info['variance_explained'] = var_explained
    info['t_pca'] = time.time() - t1
    print(f"  PCA({n_pca}): {var_explained:.1%} variance, {info['t_pca']:.1f}s")

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
    parser = argparse.ArgumentParser(description="GPU UMAP of egocentric keypoints")
    parser.add_argument('--data-dir', required=True, help="Directory with ego_features.npy")
    parser.add_argument('--n-pca', type=int, default=20, help="PCA components before UMAP")
    parser.add_argument('--n-components', type=int, default=3, help="UMAP output dims")
    parser.add_argument('--n-neighbors', type=int, default=30,
                        help="UMAP n_neighbors (30 balances local/global)")
    parser.add_argument('--min-dist', type=float, default=0.05,
                        help="UMAP min_dist (0.05 preserves structure)")
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    # Check GPU
    import cupy as cp
    print(f"GPU: {cp.cuda.runtime.getDeviceProperties(0)['name'].decode()}")
    print(f"VRAM: {cp.cuda.runtime.memGetInfo()[1] / 1e9:.1f} GB")

    # Load preprocessed data
    print(f"\nLoading from {args.data_dir}/")
    features = np.load(os.path.join(args.data_dir, 'ego_features.npy'))
    valid = np.load(os.path.join(args.data_dir, 'valid_mask.npy'))
    meta = dict(np.load(os.path.join(args.data_dir, 'metadata.npz'), allow_pickle=True))
    print(f"  Features: {features.shape} ({features.nbytes/1e6:.0f} MB)")
    print(f"  Keypoint set: {meta.get('keypoint_set', 'unknown')}")
    print(f"  Valid: {valid.sum()}/{len(valid)} ({100*valid.sum()/len(valid):.1f}%)")

    # Filter to valid frames only
    features_valid = features[valid]
    N = features_valid.shape[0]
    print(f"  Using {N} valid frames for embedding")

    # Run UMAP
    print(f"\n{'='*60}")
    print(f"Embedding: ego ({N} × {features_valid.shape[1]} → PCA {args.n_pca} → UMAP {args.n_components}D)")
    print(f"{'='*60}")

    embedding, info = gpu_embed(
        features_valid, 'ego', args.n_pca, args.n_components,
        args.n_neighbors, args.min_dist, args.seed)

    if embedding is None:
        print(f"FAILED: {info.get('error', 'unknown')}")
        return

    # Expand back to full frame count (NaN for invalid frames)
    full_embedding = np.full((len(valid), args.n_components), np.nan, dtype=np.float32)
    full_embedding[valid] = embedding

    out_path = os.path.join(args.data_dir, 'umap_ego.npy')
    np.save(out_path, full_embedding)
    print(f"\nSaved: {out_path} ({full_embedding.nbytes/1e6:.0f} MB)")

    # Save params
    params = {
        'n_frames_total': int(len(valid)),
        'n_frames_valid': int(N),
        'n_pca': args.n_pca,
        'n_components': args.n_components,
        'n_neighbors': args.n_neighbors,
        'min_dist': args.min_dist,
        'seed': args.seed,
        'keypoint_set': str(meta.get('keypoint_set', 'unknown')),
        'embedding': info,
    }
    with open(os.path.join(args.data_dir, 'umap_params.json'), 'w') as f:
        json.dump(params, f, indent=2)

    print(f"\nDone: PCA {info['variance_explained']:.1%} var, "
          f"UMAP {info['t_umap']:.0f}s, total {info['t_total']:.0f}s")


if __name__ == "__main__":
    main()
