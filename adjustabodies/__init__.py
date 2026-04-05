"""adjustabodies — MuJoCo body model fitting for behavioral neuroscience.

Resize body segments, calibrate site positions (STAC), and run batch IK
for multi-camera 3D pose estimation. Supports rodent, mouse, and fly models.

Usage:
    from adjustabodies import load_model, fit_body_model, batch_ik
    from adjustabodies.species import rodent
"""

__version__ = "0.1.0"


def enable_jax_cache():
    """Enable persistent JAX compilation cache (saves ~2 min per job).

    Cache dir is version-specific so JAX upgrades don't use stale binaries.
    Safe to call multiple times (idempotent).
    """
    try:
        import jax
        import os
        cache_dir = os.path.expanduser(f"~/.jax_cache/mjx_{jax.__version__}")
        os.makedirs(cache_dir, exist_ok=True)
        jax.config.update("jax_compilation_cache_dir", cache_dir)
    except Exception:
        pass  # JAX not installed or config not supported
