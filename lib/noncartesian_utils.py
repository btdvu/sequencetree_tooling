# Utility functions for non-Cartesian trajectories.

import numpy as np
import sigpy as sp

# Compute phase adjustment to the k-space data based on the coordinates and the desired shift (in pixels).
# A positive shift corresponds to shifting the object to the right and up (in standard head-first supine coordinates with axial orientation).
#TODO: Rigorous testing of shifts and rotations with SequenceTree sequences.
#TODO: Also test multiple non-Cartesian trajectories with non-constant readout gradient amplitude sampling. (e.g. spiral, ramp-sampled radial, EPI)
#TODO: Determine whether FOV shifting needs to be done before or after gradient delay estimation. I think before, but am not sure until you try.
def shiftFieldOfView(ksp, coord, shift, img_shape=None):
    if img_shape is None:
        img_shape = sp.estimate_shape(coord)
    elif coord.shape[-1] != len(img_shape):
        raise ValueError(f"Coordinate matrix dimensions ({coord.shape[-1]}-dim) do not match image shape ({len(img_shape)}-dim)")
    elif coord.shape[-1] != len(shift):
        raise ValueError(f"Coordinate matrix dimensions ({coord.shape[-1]}-dim) do not match shift dimensions ({len(shift)}-dim)")

    phase_adjustment_per_direction = coord/img_shape * 2*np.pi * shift
    
    phase_adjustment = np.sum(phase_adjustment_per_direction, axis=-1)

    adjustment = np.exp(-1j*phase_adjustment)

    ksp_adjusted = ksp*adjustment

    return ksp_adjusted    