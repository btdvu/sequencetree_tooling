"""
Echo Planar Imaging (EPI) Utility Functions

Provides functions for generating ramp-sampled EPI trajectories and performing 
gradient-delay induced ghosting correction using calibration data.

Author: Brian-Tinh Vu
Date: 04/06/2026
Dependencies: numpy, sigpy

Quick Start:
    traj = trajFullSampling(N_plat=128, N_ramp=32, nx=128, ny=128)
    delta_k = estGradDelay(ksp_calib, N_plat=128, nx=128)
    traj_corr = shiftTrajectory(traj, delta_k)
"""

import numpy as np
import sigpy as sp
import lib.ramp_sampling
import lib.radial


def trajFullSampling(N_plat, N_ramp, nx, ny, negative_polarity=False):
    """
    Generates a fully-sampled EPI trajectory with ramp sampling.
    
    Creates a Cartesian k-space grid sampled along zig-zag pathways in the readout direction.
    It accounts for spatial sampling during the ramp portions of gradient blips.

    Gradient delays are estimated using the cross-correlation method presented in 
    Block KT, Uecker M. Simple method for adaptive gradient-delay compensation in radial MRI. 
    Proc 19th Annu Meeting ISMRM, Montréal, 2011, p. 2816. This method was adapted from radial 
    trajectories to the special case of EPI.
    
    Parameters
    ----------
    N_plat : int
        Number of readout samples taken during the plateau of the readout gradient.
    N_ramp : int
        Number of readout samples taken during the gradient ramp-up/down.
    nx : int
        Number of points in the target image grid along the readout direction.
    ny : int
        Number of points in the target image grid along the phase encode direction.
    negative_polarity : bool, optional
        Whether the first readout line is acquired in the negative k-space direction. Default: False.
        
    Returns
    -------
    coord : np.ndarray
        EPI k-space coordinates including ramp-sampling density. shape: (ny, nx_samples, 2)
    """
    y_lines = np.arange(-ny//2, -ny//2+ny)
    # 1D readout line taking ramp sampling into account
    x_coord = lib.ramp_sampling.readoutLine(N_plat, N_ramp, nx)

    # Invert readout direction if starting with negative polarity
    if negative_polarity:
        x_coord *= -1

    coord = []
    # Build 2D trajectory line-by-line, alternating readout direction for EPI zig-zag
    for i, y_coord in enumerate(y_lines):
        line_coord = np.stack([x_coord*(-1)**(i%2), np.ones_like(x_coord)*y_coord], axis=-1)
        coord.append(line_coord)
    coord = np.array(coord)

    return coord


def shiftTrajectory(coord_nom, delta_k, negative_polarity=False):
    """
    Applies measured shifts in k-space to mitigate EPI ghosting.
    
    Subtracts measured 1D shifts from the nominal trajectory to align odd and even
    readouts, thus removing Nyquist ghosts without interpolating the data directly.
    
    Parameters
    ----------
    coord_nom : np.ndarray
        Nominal uncorrected k-space trajectory. shape: (n_lines, n_samples, 2)
    delta_k : np.ndarray
        1D array of shift amounts per readout line. shape: (n_lines,)
    negative_polarity : bool, optional
        Set to True if the trajectory begins with a negative polarity readout. Default: False.
        
    Returns
    -------
    coord_corr : np.ndarray
        Shift-corrected k-space coordinates. shape: (n_lines, n_samples, 2)
    """
    # Shift only happens along the readout (x) direction
    delta_k_pad = np.stack([delta_k, np.zeros_like(delta_k)], axis=-1)

    # Add singleton dimension to broadcast across the readout samples
    delta_k_toBeAdded = np.reshape(delta_k_pad, (delta_k_pad.shape[0], 1, delta_k_pad.shape[1]))

    if negative_polarity:
        delta_k_toBeAdded *= -1

    # Correct the trajectory by applying the estimated delay shifts
    coord_corr = coord_nom - delta_k_toBeAdded

    return coord_corr


def estGradDelayMultiCh(ksp_calib, N_plat, nx, thresh=0.1):
    """
    Estimates the gradient-delay induced shift for multi-channel EPI data.
    
    Computes k-space shifts independently for each coil element and then performs
    an L2-norm weighted algebraic combination across coils.

    L2-norm weighted combination of signal is common across MRI literature and is used for gradient measurements in 
    P. T. Gurney, “Magnetic resonance imaging using a 3D cones k-space trajectory,” Ph.D. dissertation, 
    Dept. Electr. Eng., Stanford Univ., Stanford, CA, 2007.
    
    Parameters
    ----------
    ksp_calib : np.ndarray
        Calibration k-space data (e.g. from non-phase encoded navigators 
        acquired with positive and negative polarities). shape: (2, n_coils, n_pairs, n_samples)
    N_plat : int
        Number of readout samples taken during the plateau of the readout gradient.
    nx : int
        Number of points in the target image grid along the readout direction.
    thresh : float, optional
        Threshold used to mask the signal, removing noise dominance in phase slope estimation. Default: 0.1.
        
    Returns
    -------
    delta_k : np.ndarray
        Estimated shift values in k-space units for each readout pair across the array. shape: (n_pairs,)
    """
    # extract parameters
    n_coils = ksp_calib.shape[1]
    n_pairs = ksp_calib.shape[2]

    # Calculate coil weights using the L2-norm to favor elements with higher SNR
    coil_weights = np.zeros(n_coils)
    for i_coil in range(n_coils):
        coil_weights[i_coil] = np.linalg.norm(ksp_calib[:,i_coil])
    
    coil_weights = coil_weights/np.sum(coil_weights)

    # compute delta_k shifts for each coil
    delta_k_per_coil = np.zeros((n_coils, n_pairs))
    for i_coil in range(n_coils):
        delta_k_per_coil[i_coil] = estGradDelay(ksp_calib[:,i_coil], N_plat, nx, thresh=thresh)

    # weighted sum of delta_k shifts
    delta_k = np.zeros(n_pairs)
    for i_coil in range(n_coils):
        delta_k += coil_weights[i_coil] * delta_k_per_coil[i_coil]

    return delta_k


#TODO: Output 'delta_k' may have a shape of (n_pairs,), not (n_pairs * 2,)
def estGradDelay(ksp_calib, N_plat, nx, thresh=0.1):
    """
    Estimates the gradient-delay induced shift from EPI positive/negative polarities.
    
    Uses phase difference between odd and even readouts in image space to compute a spatial
    shift, which translates to a linear phase slope corresponding to the k-space shift 
    causing Nyquist ghosting.
    
    Parameters
    ----------
    ksp_calib : np.ndarray
        Single-channel calibration k-space data. shape: (2, n_pairs, n_samples)
    N_plat : int
        Number of readout samples taken during the gradient plateau.
    nx : int
        Number of target grid points along the readout axis.
    thresh : float, optional
        Fractional threshold applied down from the peak to mask noise regions. Default: 0.1.
        
    Returns
    -------
    delta_k : np.ndarray
        Estimated k-space shifts for the readouts. shape: (n_pairs * 2,)
    """
    # extract parameters
    N_ro = ksp_calib.shape[2]
    N_ramp = (N_ro - N_plat)//2
    n_pairs = ksp_calib.shape[1]

    # get the plateau portion of each readout
    ksp_plat = sp.resize(ksp_calib, oshape=(2, n_pairs, N_plat))
    
    # Isolate parallel and antiparallel (idx 0) readouts
    # We reference shifts to the parallel readout direction
    par_ro = ksp_plat[1]
    antipar_ro = ksp_plat[0]

    # Flip antiparallel readouts to physically match the parallel lines in k-space
    antipar_ro_flip = np.flip(antipar_ro, axis=-1)

    # apply IFFT along readout direction
    par_ro_f = sp.ifft(par_ro, axes=(-1,))
    antipar_ro_flip_f = sp.ifft(antipar_ro_flip, axes=(-1,))

    # complex conjugate of antiparallel readouts
    antipar_ro_flip_f_conj = np.conj(antipar_ro_flip_f)

    # Translate to cross-correlation to find phase discrepancy
    # Phase slope here represents sub-pixel shifts between the two readout directions
    g_r = par_ro_f * antipar_ro_flip_f_conj

    # Build mask to ignore phase estimation in noise-dominated background regions
    mask = np.zeros((n_pairs, N_plat), dtype=np.int32)

    for i_ro in range(n_pairs):
        tmp_ro = par_ro[i_ro]
        mask[i_ro] = lib.radial._findSpokeSupport(tmp_ro, thresh=thresh)

    # apply mask to cross-correlation
    g_r = mask * g_r

    # Compute phase slopes using linear regression
    slope_collector = np.zeros(n_pairs)

    for i_spoke in range(0, n_pairs):
        g_r_cropped = np.squeeze(g_r[i_spoke, np.nonzero(mask[i_spoke])])
        poly_coeffs = np.polyfit(np.arange(0, len(g_r_cropped)), np.unwrap(np.angle(g_r_cropped)), 1)
        slope_collector[i_spoke] = poly_coeffs[0]

    # Shift is proportional to slope. The 1/2 factor accounts for the opposing readout polarities.
    delta_k_unscaled = (1/2)*slope_collector*N_plat/(2*np.pi)

    # Compute oversampling factor because ramp points contribute slightly less spatial field-of-view progression
    osf = (N_ramp + N_plat)/nx

    # Normalize shifts by the oversampling factor
    delta_k = delta_k_unscaled/osf

    # flip odd-indexed readouts
    for i in range(len(delta_k)):
        if i%2 == 1:
            delta_k[i] *= -1

    return delta_k