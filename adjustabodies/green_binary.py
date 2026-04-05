"""Reader for Green's trajectory binary format (green_traj3d.bin).

Shared parser used by fitting, IK, and UMAP scripts. Avoids duplicating
the binary parsing code across multiple scripts.
"""

import struct
import numpy as np
from typing import List, Tuple, Optional


MAGIC = 0x024E5247  # "GRN\x02"


def load_green_binary(path: str):
    """Load a Green binary file's index and field info.

    Args:
        path: path to .bin file (green_traj3d.bin or repaired_traj3d.bin)

    Returns:
        data: np.memmap (uint8) of the entire file
        trials_index: list of (byte_offset, num_frames) per trial
        traj3d_field: dict with 'name', 'epf', 'esz', 'offset' for traj3d
    """
    data = np.memmap(path, dtype=np.uint8, mode='r')
    magic = struct.unpack_from('<I', data, 0)[0]
    assert magic == MAGIC, f"Bad magic: {magic:#x}"

    num_trials = struct.unpack_from('<I', data, 8)[0]
    num_fields = struct.unpack_from('<I', data, 12)[0]

    fields = []
    cumulative = 0
    for i in range(num_fields):
        pos = 32 + i * 44
        name = bytes(data[pos:pos+32]).split(b'\0')[0].decode()
        epf = struct.unpack_from('<I', data, pos + 32)[0]
        esz = struct.unpack_from('<I', data, pos + 36)[0]
        fields.append({'name': name, 'epf': epf, 'esz': esz, 'offset': cumulative})
        cumulative += epf * esz

    traj3d = next(f for f in fields if f['name'] == 'traj3d')

    desc_end = 32 + num_fields * 44
    index_start = (desc_end + 7) & ~7

    trials_index = []
    for i in range(num_trials):
        pos = index_start + i * 12
        offset = struct.unpack_from('<Q', data, pos)[0]
        nf = struct.unpack_from('<I', data, pos + 8)[0]
        trials_index.append((offset, nf))

    return data, trials_index, traj3d


def read_trial_keypoints(data, trials_index, traj3d, trial_id: int,
                          frame_start: Optional[int] = None,
                          frame_end: Optional[int] = None) -> np.ndarray:
    """Read 3D keypoints for one trial.

    Args:
        data: memmap from load_green_binary
        trials_index: trial index from load_green_binary
        traj3d: traj3d field info from load_green_binary
        trial_id: trial index (0-based)
        frame_start: optional start frame (inclusive)
        frame_end: optional end frame (inclusive)

    Returns:
        kp3d: [N, 24, 3] float32 array in mm (first 24 keypoints, excludes ball)
    """
    offset, nf = trials_index[trial_id]
    stride = traj3d['epf']
    traj_offset = traj3d['offset']

    start = offset + nf * traj_offset
    nbytes = nf * stride * 4
    arr = np.frombuffer(data[start:start + nbytes], dtype=np.float32).reshape(nf, stride)
    kp3d = arr[:, :72].reshape(nf, 24, 3)

    if frame_start is not None or frame_end is not None:
        fs = frame_start or 0
        fe = (frame_end or nf - 1) + 1
        kp3d = kp3d[fs:fe]

    return kp3d
