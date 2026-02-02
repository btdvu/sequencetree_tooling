# Functions for EPI acquisition.
# All EPI trajectories are ramp sampled.

import numpy as np
import sigpy as sp
import lib.ramp_sampling
import lib.radial

# Generates a fully-sampled EPI trajectory.
def trajFullSampling(N_plat, N_ramp, nx, ny, negative_polarity=False):
    y_lines = np.arange(-ny//2, -ny//2+ny)
    x_coord = lib.ramp_sampling.readoutLine(N_plat, N_ramp, nx)

    if negative_polarity:
        x_coord *= -1

    coord = []
    for i, y_coord in enumerate(y_lines):
        line_coord = np.stack([x_coord*(-1)**(i%2), np.ones_like(x_coord)*y_coord], axis=-1)
        coord.append(line_coord)
    coord = np.array(coord)

    return coord


# Applies the measured shifts in k-space to mitigate EPI ghosting.
def shiftTrajectory(coord_nom, delta_k, negative_polarity=False):
    # pad with zeros for the y-coordinate
    delta_k_pad = np.stack([delta_k, np.zeros_like(delta_k)], axis=-1)

    # add singleton dimension to broadcast with readout dimension
    delta_k_toBeAdded = np.reshape(delta_k_pad, (delta_k_pad.shape[0], 1, delta_k_pad.shape[1]))

    if negative_polarity:
        delta_k_toBeAdded *= -1

    # correct the trajectory
    coord_corr = coord_nom - delta_k_toBeAdded

    return coord_corr




# Estimates the gradient-delay induced shift for each readout line of an EPI, given EPI data with positive and negative polarity.
# This version handles multi-channel coil data.
def estGradDelayMultiCh(ksp_calib, N_plat, nx, thresh=0.1):
    # extract parameters
    n_coils = ksp_calib.shape[1]
    n_pairs = ksp_calib.shape[2]


    # compute coil weights by L2-norm
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


# Estimates the gradient-delay induced shift for each readout line of an EPI, given EPI data with positive and negative polarity.
def estGradDelay(ksp_calib, N_plat, nx, thresh=0.1):
    # extract parameters
    N_ro = ksp_calib.shape[2]
    N_ramp = (N_ro - N_plat)//2
    n_pairs = ksp_calib.shape[1]

    # get the plateau portion of each readout
    ksp_plat = sp.resize(ksp_calib, oshape=(2, n_pairs, N_plat))
    # separate parallel (idx 1) and antiparallel (idx 0) readouts
    # parallel readout direction is the convention for correction
    par_ro = ksp_plat[1]
    antipar_ro = ksp_plat[0]

    # flip antiparallel readouts
    antipar_ro_flip = np.flip(antipar_ro, axis=-1)

    # apply IFFT along readout direction
    par_ro_f = sp.ifft(par_ro, axes=(-1,))
    antipar_ro_flip_f = sp.ifft(antipar_ro_flip, axes=(-1,))

    # complex conjugate of antiparallel readouts
    antipar_ro_flip_f_conj = np.conj(antipar_ro_flip_f)

    # cross-correlation function in Fourier domain
    g_r = par_ro_f * antipar_ro_flip_f_conj

    # initialize mask for each readout pair
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

    # Compute k-space shifts; this is without scaling for the oversampling factor
    delta_k_unscaled = (1/2)*slope_collector*N_plat/(2*np.pi)

    # compute the oversampling factor
    osf = (N_ramp + N_plat)/nx

    # Scale the k-space shifts
    delta_k = delta_k_unscaled/osf

    # flip odd-indexed readouts
    for i in range(len(delta_k)):
        if i%2 == 1:
            delta_k[i] *= -1

    return delta_k