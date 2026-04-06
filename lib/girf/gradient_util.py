"""
Gradient and K-Space Utilities

Provides foundational linear algebraic mappings between non-dimensional k-space
and physical gradient zeroth moments, matching the internal scaling and units of SequenceTree.

k-space coordinates are expressed in units of 1/FOV, which is standard in modern NUFFT libraries.

Typically, this means that for a 128x128 image, the k-space coordinates range from -64 to +64 in both kx and ky.

Author: Brian-Tinh Vu
Date: 04/06/2026
Dependencies: numpy

Quick Start:
    moments = kspace2moment(kspace_target, fov=[256, 256, 256])
    loc = moment2kspace(accumulated_moment, fov=[256, 256, 256])
"""

import numpy as np


def kspace2moment(kspace, fov, gamma=42.5764):
    """
    Computes required gradient zeroth moments derived from a target k-space location.
    
    Transforms the normalized (dimensionless) k-space trajectory coordinate 
    into physical gradient moments ([uT/mm]-us) required to reach that point,
    scaled by the encoding field-of-view and the gyromagnetic ratio.
    
    Parameters
    ----------
    kspace : np.ndarray
        Coordinates in target k-space. shape: (3, n_points) where 3 is [x, y, z].
    fov : list or np.ndarray
        Field-of-view defining the imaging volume in mm. shape: (3,)
    gamma : float, optional
        Gyromagnetic ratio in MHz/T. Default: 42.5764 (Proton).
        
    Returns
    -------
    moments : np.ndarray
        Gradient zeroth moments corresponding to the k-space path along each physical axis.
        shape: (3, n_points)
    """
    tmp_fov = np.reshape(fov, [3] + (len(kspace.shape)-1)*[1])
    return kspace/(tmp_fov * gamma)*1E6


def moment2kspace(moment, fov, gamma=42.5764):
    """
    Computes k-space trajectory locations from accumulated physical gradient moments.
    
    Inverts the moment-to-kspace relationship to determine the actual k-space
    traversed given a physical accumulation of gradient zeroth moments over time.
    
    Parameters
    ----------
    moment : np.ndarray
        Accumulated zeroth moments along each axis in [uT/mm]-us. shape: (3, n_points)
    fov : list or np.ndarray
        Field-of-view defining the imaging volume in mm. shape: (3,)
    gamma : float, optional
        Gyromagnetic ratio in MHz/T. Default: 42.5764 (Proton).
        
    Returns
    -------
    kspace_coords : np.ndarray
        Tracked non-dimensional coordinates in k-space. shape: (3, n_points)
    """
    tmp_fov = np.reshape(fov, [3] + (len(moment.shape) - 1) * [1])
    return moment*(tmp_fov * gamma)/1E6



# # Computes a gradient waveform based on an array of moments, assuming the moments are spaced 10 us apart.
# # NOTE: This implementation is identical to how gradients are computed in SequenceTree!
# # Args:
#     # 'moment_10': array of moments along each axis; [x/y/z, n_steps+1, m1, ..., mD]; [uT/mm]-us
#         # 'n_steps': number of 10 us intervals in 'plateau_time', the duration of the arbitrary gradient
# # Returns:
#     # array of gradient amplitudes; [x/y/z, n_steps, m1, ..., mD]; uT/mm
# def moment2gradient(moment_10):
#     n_steps = moment_10.shape[1]-1
#     moment_10_plus10 = np.roll(moment_10, -1, axis=1)
#     gradient_10 = (moment_10_plus10 - moment_10)/10
#     return gradient_10[:, :n_steps]
#
#
# # Computes accumulated moment based on a gradient waveform, assuming the gradient values are spaced 10 us apart.
# # NOTE: This implementation is identical to how moments are computed in SequenceTree!
# # Args:
#     # 'gradient_10': array of gradients along each axis; [x/y/z, n_steps, m1, ..., mD]; [uT/mm]-us
#         # 'n_steps': number of 10 us intervals in 'plateau_time', the duration of the arbitrary gradient
# # Returns:
#     # array of moments; [x/y/z, n_steps, m1, ..., mD]; uT/mm
# def gradient2moment(gradient_10):
#     moment_10 = np.cumsum(gradient_10*10, axis=1)
#     return moment_10