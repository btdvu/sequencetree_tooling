"""
Ramp Sampling Trajectory Generation for MRI

Generate radial k-space trajectories with ramp sampling on trapezoidal gradients.
Samples k-space during both gradient ramps and plateau for improved efficiency.
Supports golden-angle sampling, stack-of-stars 3D acquisition, and gradient delay correction.

Author: Brian-Tinh Vu
Date: 01/26/2026
Dependencies: numpy, sigpy

Quick Start:
    traj = trajGoldenAngle(N_plat=128, N_ramp=32, shape_i=128, n_spokes=360)
    traj_3d = trajGoldenAngleStackOfStars(N_plat=128, N_ramp=32, shape_i=128, n_spokes_per_slice=360, nz=64)
    shift_matrix = estGradDelay(ksp_calib, N_plat=128, N_ramp=32, shape_i=128)
"""

import numpy as np
import lib.radial
import sigpy as sp

def estGradDelay(ksp_calib, N_plat, N_ramp, shape_i):
    """
    Estimate gradient delays from ramp-sampled calibration data.
    
    Extracts plateau region from ramp-sampled data and estimates gradient
    delay shift matrix, then rescales to account for oversampling factor.
    
    Parameters
    ----------
    ksp_calib : np.ndarray
        Multi-channel calibration k-space data shape: (n_coils, n_spokes, n_ro).
        Must contain ramp-sampled parallel/antiparallel spoke pairs.
    N_plat : int
        Number of samples on gradient plateau.
    N_ramp : int
        Number of samples on each gradient ramp.
    shape_i : int
        Target image matrix size.
    
    Returns
    -------
    shift_matrix : np.ndarray
        2x2 gradient delay shift matrix in scaled k-space coordinates.
    
    Notes
    -----
    Uses plateau-only data for delay estimation via lib.radial.estGradDelayMultiCh.
    Accounts for oversampling factor: osf = (N_ramp + N_plat) / shape_i.
    """
    n_coils = ksp_calib.shape[0]
    n_calibPairs = ksp_calib.shape[1]//2

    # Get the plateau of the calibration spokes
    ksp_plat = sp.resize(ksp_calib, (n_coils, 2*n_calibPairs, N_plat))
    
    # Get the unscaled trajectory of the calibration data
    coord_plat = lib.radial.trajCalibration(N_plat, n_calibPairs)

    # Estimate the unscaled shift matrix
    shift_matrix_unscaled = lib.radial.estGradDelayMultiCh(ksp_plat, coord_plat)

    # Compute the oversampling factor on the plateau
    osf = (N_ramp + N_plat)/shape_i

    # Shift matrix in scaled k-space coordinates
    shift_matrix = shift_matrix_unscaled/osf

    return shift_matrix


def trajGoldenAngleStackOfStars(N_plat, N_ramp, shape_i, n_spokes_per_slice, nz, staggered=False):
    """
    Generate 3D stack-of-stars trajectory with ramp-sampled golden-angle radial.
    
    Creates a 3D k-space trajectory by stacking 2D ramp-sampled golden-angle
    radial trajectories along the z-axis. Each slice contains a full set of
    radial spokes with ramp sampling, and slices can be staggered for improved coverage.
    
    Parameters
    ----------
    N_plat : int
        Number of samples on gradient plateau.
    N_ramp : int
        Number of samples on each gradient ramp.
    shape_i : int
        Target image matrix size.
    n_spokes_per_slice : int
        Number of radial spokes per slice.
    nz : int
        Number of slices along the z-axis.
    staggered : bool, optional
        If True, stagger starting angles between slices for improved coverage.
        Default: False.
    
    Returns
    -------
    traj : np.ndarray
        3D k-space coordinates shape: (nz*n_spokes_per_slice, N_plat+2*N_ramp, 3).
        Contains (kx, ky, kz) coordinates for each spoke.
    
    Notes
    -----
    - Uses 111.25° golden-angle increment for optimal xy-plane coverage
    - z-coordinates are centered around zero (symmetric sampling)
    - When staggered=True, each slice starts at a different angle to
      improve 3D coverage and reduce coherent aliasing
    - Total readout length per spoke: N_ro = N_plat + 2*N_ramp
    """
    # Generate z-coordinates for symmetric slice coverage
    pez_lines = np.arange(-nz//2, -nz//2+nz)

    # Initialize list to store trajectory for each slice
    slice_trajectories = []

    # Generate radial trajectory for each slice
    for i, pez in enumerate(pez_lines):
        # Determine starting angle for this slice
        if staggered:
            # Stagger angles between slices using golden-angle offset
            starting_angle = 111.25*i*n_spokes_per_slice
        else:
            # All slices start at same angle
            starting_angle = 0

        # Generate 2D radial trajectory for this slice
        tmp_slice_traj = trajGoldenAngle(N_plat, N_ramp, shape_i, n_spokes_per_slice, starting_angle=starting_angle)

        # Add z-coordinate to create 3D trajectory
        # Broadcast z-coordinate to match trajectory shape
        z_coords = np.ones(list(tmp_slice_traj.shape[:2])+[1]) * pez
        tmp_slice_traj_3d = np.concatenate([tmp_slice_traj, z_coords], axis=-1)

        slice_trajectories.append(tmp_slice_traj_3d)
    
    # Concatenate all slices into single 3D trajectory
    slice_trajectories = np.concatenate(slice_trajectories, axis=0)

    return slice_trajectories




def trajGoldenAngle(N_plat, N_ramp, shape_i, n_spokes, starting_angle=0, negative_angle=False):
    """
    Generate golden-angle radial trajectory with ramp sampling.
    
    Creates 2D radial k-space trajectory that samples during both gradient
    ramps and plateau of trapezoidal readout gradients for improved efficiency.
    
    Parameters
    ----------
    N_plat : int
        Number of samples on gradient plateau.
    N_ramp : int
        Number of samples on each gradient ramp.
    shape_i : int
        Target image matrix size.
    n_spokes : int
        Number of radial spokes (projections).
    starting_angle : float, optional
        Angle of the first golden-angle spoke in degrees. Default: 0.
    negative_angle : bool, optional
        If True, invert trajectory direction. Default: False.
    
    Returns
    -------
    traj : np.ndarray
        K-space coordinates shape: (n_spokes, N_plat+2*N_ramp, 2).
        Contains (kx, ky) coordinates for each spoke.
    
    Notes
    -----
    Uses 111.25° golden-angle increment for optimal coverage.
    Total readout length: N_ro = N_plat + 2*N_ramp.
    K-space positions account for gradient amplitude variation during ramps.
    """
    N_ro = N_plat + 2*N_ramp

    # Initialize trajectory array
    traj = np.zeros((n_spokes, N_ro, 2), dtype=float)

    # Golden-angle increment (degrees)
    PI = 3.141592
    angle_inc = 111.25

    # Compute radial positions along spoke
    line = readoutLine(N_plat, N_ramp, shape_i)

    # Generate each radial spoke
    for i_projection in range(n_spokes):
        # Calculate spoke angle
        angle_in_deg = i_projection * angle_inc + starting_angle
        angle = angle_in_deg * PI / 180.0

        # Direction vector for this spoke
        kx_dir = np.cos(angle)
        ky_dir = np.sin(angle)

        if negative_angle:
            kx_dir *= -1
            ky_dir *= -1
        
        # Scale radial positions by direction
        line_x = kx_dir * line
        line_y = ky_dir * line

        # Store coordinates
        traj[i_projection, :, 0] = line_x
        traj[i_projection, :, 1] = line_y
    
    return traj


def readoutLine(N_plat, N_ramp, shape_i):
    """
    Compute 1D k-space sampling positions for ramp-sampled trapezoidal gradient.
    
    Calculates k-space locations along a single readout dimension accounting
    for gradient amplitude variation during ramps and plateau.
    
    Parameters
    ----------
    N_plat : int
        Number of samples on gradient plateau.
    N_ramp : int
        Number of samples on each gradient ramp.
    shape_i : int
        Target image matrix size.
    
    Returns
    -------
    line : np.ndarray
        1D k-space positions shape: (N_plat+2*N_ramp,).
        Centered at zero with scaled coordinates.
    
    Notes
    -----
    Oversampling factor: osf = (N_ramp + N_plat) / shape_i.
    K-space positions are rescaled to match target image matrix size.
    """
    # Compute oversampling factor on the plateau
    # Derived by equating gradient moments between standard and ramp-sampled readouts
    osf = (N_ramp + N_plat)/shape_i

    line = _line(N_plat, N_ramp, osf)

    return line


def _line(N_plat, N_ramp, osf):
    """
    Compute k-space trajectory for trapezoidal gradient with ramp sampling.
    
    Parameters
    ----------
    N_plat : int
        Number of samples on gradient plateau.
    N_ramp : int
        Number of samples on each gradient ramp.
    osf : float
        Oversampling factor.
    
    Returns
    -------
    k_space_moment_rescaled : np.ndarray
        Rescaled k-space positions centered at zero.
    
    Notes
    -----
    Gradient moment calculation uses left-sided Riemann sum.
    Trapezoidal gradient has linear ramps and constant plateau.
    """
    # Create the ramp-up segment
    sampling_pts_rampUp = np.arange(0, N_ramp, 1)
    # Gradient values at each point during ramp up
    grad_ampl_rampUp = sampling_pts_rampUp/N_ramp

    # Create the plateau segment
    sampling_pts_plat = np.arange(N_ramp, N_ramp+N_plat, 1)
    grad_ampl_plat = np.ones_like(sampling_pts_plat)

    # Create the ramp-down segment
    sampling_pts_rampDown = np.arange(N_ramp+N_plat, 2*N_ramp+N_plat, 1)
    grad_ampl_rampDown = 1 - (sampling_pts_rampDown - (N_ramp+N_plat))/N_ramp

    # Create the trapezoid gradient
    grad_trapezoid = np.concatenate([grad_ampl_rampUp, grad_ampl_plat, grad_ampl_rampDown])
    # Compute the k-space moment from the trapezoid gradient
    k_space_moment = np.cumsum(grad_trapezoid)

    # Find the central k-space point
    if len(k_space_moment)%2 == 0:
        idx_center = len(k_space_moment)//2 - 1
        k_space_moment = k_space_moment - k_space_moment[idx_center] - 0.5
    else:
        idx_center = len(k_space_moment)//2
        k_space_moment = k_space_moment - k_space_moment[idx_center]

    # Rescale the k_space_moment
    k_space_moment_rescaled = k_space_moment/osf

    return k_space_moment_rescaled


