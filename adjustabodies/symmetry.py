"""Bilateral symmetry enforcement for site offsets.

Matches RED's C++ STAC implementation (mujoco_stac.h):
  - Midline sites: Y offset forced to 0
  - L/R pairs: average X/Z offsets, mirror Y offsets
"""

import numpy as np
from typing import List, Tuple

try:
    import jax.numpy as jnp
    HAS_JAX = True
except ImportError:
    HAS_JAX = False


def build_symmetry_indices_np(m, site_pairs, midline_sites):
    """Build symmetry index arrays (numpy) from a compiled MuJoCo model.

    Returns: (midline_ids, pair_L_ids, pair_R_ids) as numpy int arrays.
    """
    site_name_to_id = {}
    for i in range(m.nsite):
        site_name_to_id[m.site(i).name] = i

    midline_ids = [site_name_to_id[n] for n in midline_sites if n in site_name_to_id]
    pair_L, pair_R = [], []
    for ln, rn in site_pairs:
        li = site_name_to_id.get(ln, -1)
        ri = site_name_to_id.get(rn, -1)
        if li >= 0 and ri >= 0:
            pair_L.append(li)
            pair_R.append(ri)

    return (np.array(midline_ids, dtype=np.int32),
            np.array(pair_L, dtype=np.int32),
            np.array(pair_R, dtype=np.int32))


def enforce_symmetry_np(offsets, midline_ids, pair_L_ids, pair_R_ids):
    """Enforce bilateral symmetry on site offsets (numpy, in-place)."""
    offsets[midline_ids, 1] = 0.0

    avg_x = (offsets[pair_L_ids, 0] + offsets[pair_R_ids, 0]) * 0.5
    avg_z = (offsets[pair_L_ids, 2] + offsets[pair_R_ids, 2]) * 0.5
    avg_y = (offsets[pair_L_ids, 1] - offsets[pair_R_ids, 1]) * 0.5

    offsets[pair_L_ids, 0] = avg_x
    offsets[pair_L_ids, 1] = avg_y
    offsets[pair_L_ids, 2] = avg_z
    offsets[pair_R_ids, 0] = avg_x
    offsets[pair_R_ids, 1] = -avg_y
    offsets[pair_R_ids, 2] = avg_z
    return offsets


if HAS_JAX:
    def build_symmetry_indices_jax(m, site_pairs, midline_sites):
        """Build JAX-compatible symmetry index arrays."""
        mid, pL, pR = build_symmetry_indices_np(m, site_pairs, midline_sites)
        return jnp.array(mid), jnp.array(pL), jnp.array(pR)

    def enforce_symmetry_jax(offsets, midline_ids, pair_L_ids, pair_R_ids):
        """Enforce bilateral symmetry on site offsets (JAX, returns new array)."""
        offsets = offsets.at[midline_ids, 1].set(0.0)

        avg_x = (offsets[pair_L_ids, 0] + offsets[pair_R_ids, 0]) * 0.5
        avg_z = (offsets[pair_L_ids, 2] + offsets[pair_R_ids, 2]) * 0.5
        avg_y = (offsets[pair_L_ids, 1] - offsets[pair_R_ids, 1]) * 0.5

        offsets = offsets.at[pair_L_ids, 0].set(avg_x)
        offsets = offsets.at[pair_L_ids, 1].set(avg_y)
        offsets = offsets.at[pair_L_ids, 2].set(avg_z)
        offsets = offsets.at[pair_R_ids, 0].set(avg_x)
        offsets = offsets.at[pair_R_ids, 1].set(-avg_y)
        offsets = offsets.at[pair_R_ids, 2].set(avg_z)
        return offsets
