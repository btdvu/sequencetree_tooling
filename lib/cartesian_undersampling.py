"""
cartesian_undersampling.py

Functions to perform Cartesian undersampling of phase-encode lines
in 2D and 3D MRI acquisitions.

Author: Brian-Tinh Vu
Date: 11/08/2025

Requirements:
    numpy

Overview:
    - regular_undersampling: Generates a binary mask for regular or golden-angle 
      undersampling in 2D or 3D Cartesian k-space. Includes calibration region.
    - compute_accel: Utility to compute actual acceleration from the mask (undersampling factor).

Usage:
    mask = regular_undersampling(img_shape, accel, calib=24, mode="2D")
    accel_measured = compute_accel(mask)
"""

import numpy as np
import copy

def regular_undersampling(img_shape, accel, calib=24, mode="2D", tol=0.1, max_attempts=100):
    """
    Generate undersampling masks for 2D or 3D MRI Cartesian acquisitions.

    Args:
        img_shape (tuple): Shape of image (for 2D: (Nx, Ny), for 3D: (Ny, Nz)).
        accel (float): Desired acceleration factor (total lines / scanned lines).
        calib (int): Size of calibration region in PE direction(s).
        mode (str): '2D' or '3D'. Determines the undersampling scheme.
        tol (float): Acceleration tolerance (mask will be within accel ± tol).
        max_attempts (int): Maximum number of attempts to reach desired acceleration.

    Returns:
        mask (ndarray): Binary mask with sampled locations set to 1 and others 0.
                         Shape: (Nx, Ny) for 2D, (Ny, Nz) for 3D.

    Raises:
        ValueError: If mode is invalid/not supported.
    """

    if mode == "2D":
        # Get spatial dimensions
        Nx = img_shape[0]
        Ny = img_shape[1]

        # Initialize a temporary mask for iterations
        tmp_mask = np.zeros(Ny, dtype=np.complex64)

        # Track current acceleration of sampling
        actual_accel = -1

        # Start increment estimate for PE lines sampling
        test_increment = accel

        # Search bounds for increment (to approach desired accel)
        inc_upper_bound = test_increment * 1.5
        inc_lower_bound = test_increment * 0.5

        n_attempts = 0
        isMetTolerance = False

        # Search for increment giving closest acceleration to requested value
        while (not isMetTolerance) and (n_attempts < max_attempts):
            tmp_mask = np.zeros(Ny, dtype=np.complex64)
            n_attempts += 1

            # Define calibration region (centered)
            calib_start = Ny // 2 - calib // 2
            calib_end   = Ny // 2 + calib // 2
            tmp_mask[calib_start:calib_end] = 1

            # Sample PE lines outside calibration region at interval test_increment
            sampled_idxs = (np.round(np.arange(0, Ny-1/2, test_increment))).astype(int)
            tmp_mask[sampled_idxs] = 1

            actual_accel = compute_accel(tmp_mask)

            # Adjust increment search bounds to approach desired acceleration
            if actual_accel < accel:
                inc_lower_bound = test_increment
                test_increment = 0.5 * (inc_lower_bound + inc_upper_bound)
            else:
                inc_upper_bound = test_increment
                test_increment = 0.5 * (inc_lower_bound + inc_upper_bound)

            isMetTolerance = np.abs(accel - compute_accel(tmp_mask)) < tol

        # Stack mask across readout direction to produce 2D sampling mask
        mask = tmp_mask
        mask = np.stack([mask]*Nx, axis=0)
        return mask

    elif mode == "3D":
        # For 3D sampling, use two phase encode directions Ny and Nz
        Ny = img_shape[0]
        Nz = img_shape[1]
        tmp_mask = np.zeros((Ny, Nz), dtype=np.complex64)

        # Calibration region (centered square in PE plane)
        calib_start_y = Ny // 2 - calib // 2
        calib_end_y   = Ny // 2 + calib // 2
        calib_start_z = Nz // 2 - calib // 2
        calib_end_z   = Nz // 2 + calib // 2
        tmp_mask[calib_start_y:calib_end_y, calib_start_z:calib_end_z] = 1

        # Define golden means for phase encoding
        # GA_y: Golden ratio for y-direction, GA_z: irrational (sqrt(3)) for z-direction
        GA_y = (1 + np.sqrt(5)) / 2
        GA_z = np.sqrt(3)

        # Estimate how many additional PE locations are needed to achieve acceleration
        num_additional_PE = np.max([int((Ny*Nz)/accel - np.count_nonzero(tmp_mask)), 0])
        num_added_PE = 0
        n_attempts = 0

        # Iteratively fill mask until desired acceleration/tolerance met or attempts exhausted
        while (compute_accel(tmp_mask) > accel - tol) and (n_attempts < max_attempts) and (num_additional_PE > 0):
            # Indices for golden angle increments
            golden_angle_idxs = np.arange(num_added_PE, num_added_PE + num_additional_PE)

            # Compute 2D golden angle samples
            golden_angles = [(GA_y*i, GA_z*i) for i in golden_angle_idxs]
            fractional_part, integral_part = np.modf(golden_angles)

            # Map fractional parts to matrix indices
            sampled_idxs = np.array([Ny*fractional_part[:,0], Nz*fractional_part[:,1]]).astype(int)

            # Update mask with sampled indices
            tmp_mask[sampled_idxs[0], sampled_idxs[1]] = 1

            num_added_PE += num_additional_PE
            n_attempts += 1

            # Recalculate number of PE locations needed
            num_additional_PE = np.max([int((Ny*Nz)/accel - np.count_nonzero(tmp_mask)), 0])

        mask = tmp_mask
        return mask
    else:
        raise ValueError(f"Invalid mode: {mode}. Must be '2D' or '3D'.")

def compute_accel(mask):
    """
    Compute actual acceleration based on a binary mask.

    Args:
        mask (ndarray): Binary mask (sampled=1, unsampled=0).

    Returns:
        accel (float): Acceleration factor (total locations / sampled locations).
    """
    return np.prod(mask.shape) / np.count_nonzero(mask)
