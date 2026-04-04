"""adjustabodies — MuJoCo body model fitting for behavioral neuroscience.

Resize body segments, calibrate site positions (STAC), and run batch IK
for multi-camera 3D pose estimation. Supports rodent, mouse, and fly models.

Usage:
    from adjustabodies import load_model, fit_body_model, batch_ik
    from adjustabodies.species import rodent
"""

__version__ = "0.1.0"
