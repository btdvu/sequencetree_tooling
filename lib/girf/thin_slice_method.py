"""
Off-Isocenter Thin Slice Method and GIRF

Functions implementing the off-isocenter thin slice method and gradient impulse response function (GIRF).

Author: Brian-Tinh Vu
Date: 03/23/2026
Dependencies: numpy, sigpy
"""

import numpy as np
import sigpy as sp


def centerOfMass(com_meas, fov, slice_spacing, threshold=0.05):
    """
    Computes the centers of mass for a series of slices.
    
    Parameters
    ----------
    com_meas : np.ndarray
        Array of measurements from center-of-mass measurement sequence.
        Shape: (n_ro, -/+, n_slices, m1, ..., mD).
    fov : float
        Length of field-of-view (mm).
    slice_spacing : float
        Spacing between slices (mm).
    threshold : float, optional
        Value between 0 and 1 for thresholding signal and noise from slice-selection. Default: 0.05.
        
    Returns
    -------
    slice_positions : np.ndarray
        Locations of the centers of mass.
        Shape: (n_slices, m1, ..., mD).
    """
    data_com = np.copy(com_meas)
    # get number of slices
    n_slices = data_com.shape[2]
    # compute nominal slice positions
    nom_slice_pos = np.arange(-n_slices // 2, -n_slices // 2 + n_slices, 1) * slice_spacing

    # flip readouts going in the negative direction
    data_com[:, 0] = np.flip(data_com[:, 0], axis=0)
    # convert k-space to slice profiles
    img_com = sp.ifft(data_com, axes=(0,))

    # threshold readout data
    peak_values = np.max(np.abs(img_com), axis=0)
    img_com_thresh = (np.abs(img_com) > threshold*peak_values) * img_com

    # estimate the centers of mass
    n_ro = com_meas.shape[0]

    # TODO: positions are assumed to be centric; could be left-sided or right-sided
    #positions = np.linspace(-(fov / 2 - fov / n_ro / 2), fov / 2 - fov / n_ro / 2, n_ro)
    # TODO: positions are probably left sided because we use the FFT
    positions = np.arange(-fov/2, fov/2, fov/n_ro)

    # convert the thresholded slice profiles into weights
    img_weights = np.abs(img_com_thresh) / np.sum(np.abs(img_com_thresh), axis=0)
    # get the mean displacement, weighted by the signal at each position
    mean_disp = np.sum(positions * np.moveaxis(img_weights, 0, -1), axis=-1)
    # average the positive and negative displacements
    com_disp = np.mean(mean_disp, axis=0)

    # add the center-of-mass displacements to the nominal slice positions
    slice_positions = np.reshape(nom_slice_pos, [n_slices] + (len(com_disp.shape) - 1) * [1]) + com_disp

    return slice_positions


def timeEvolField(grad_meas, dwell_time, gamma=42.5764):
    """
    Computes the difference of angular derivatives (time evolution of local magnetic field).
    
    This function implements Eqn (4.17) (which is slightly incorrect) of Paul Gurney's PhD thesis at Stanford.
    
    Parameters
    ----------
    grad_meas : np.ndarray
        Array of measurements from gradient waveform measurement sequence.
        Shape: (n_ro, -/+, m1, ..., mD).
    dwell_time : float
        Spacing between sampled points in time (us).
    gamma : float, optional
        Gyromagnetic ratio (MHz/T). Default: 42.5764.
        
    Returns
    -------
    time_evol : np.ndarray
        Time evolution of local magnetic field (uT).
        Shape: (n_ro, m1, ..., mD).
    """
    data_grad = grad_meas
    n_ro = data_grad.shape[0]

    # compute right-sided derivative
    tmp_upper = np.roll(data_grad, -1, axis=0)
    tmp_lower = data_grad
    ang_deriv_rhs = ( tmp_upper * np.conj(tmp_lower) )**( 1/(2*np.pi*gamma*dwell_time) )
    # compute difference of angular derivatives
    diff_rhs = ( ang_deriv_rhs[:, 1] * np.conj(ang_deriv_rhs[:, 0]) )**(1/2)

    # compute left-sided derivative
    tmp_upper = data_grad
    tmp_lower = np.roll(data_grad, 1, axis=0)
    ang_deriv_lhs = ( tmp_upper * np.conj(tmp_lower) )**( 1/(2*np.pi*gamma*dwell_time) )
    # compute difference of angular derivatives
    diff_lhs = ( ang_deriv_lhs[:, 1] * np.conj(ang_deriv_lhs[:, 0]) )**(1/2)

    # get phase of rhs and lhs computations
    time_evol_rhs = np.angle(diff_rhs) * 1E6
    time_evol_lhs = np.angle(diff_lhs) * 1E6

    # take average of left- and right-handed estimates of the phase
    time_evol = np.zeros_like(time_evol_rhs)
    # first point uses rhs derivative only
    time_evol[0] = time_evol_rhs[0]
    # last point uses lhs derivative only
    time_evol[n_ro-1] = time_evol_lhs[n_ro-1]
    # all other points are the average of the rhs and lhs derivatives
    time_evol[1:n_ro-1] = 0.5 * (time_evol_lhs[1:n_ro-1] + time_evol_rhs[1:n_ro-1])

    return time_evol


def sampleWeights(grad_meas):
    """
    Computes the weights for least-squares estimation of the B0 and gradient waveforms.
    
    Samples are weighted by their corresponding signal magnitude.
    
    Parameters
    ----------
    grad_meas : np.ndarray
        Measurements from gradient waveform measurement sequence.
        Shape: (n_ro, -/+, m1, ..., mD).
        
    Returns
    -------
    weights : np.ndarray
        Sample weights for estimation algorithm.
        Shape: (n_ro, m1, ..., mD).
    """
    data_grad = np.copy(grad_meas)
    n_ro = data_grad.shape[0]
    # shift gradient data
    data_grad_shift = np.roll(data_grad, -1, axis=0)
    data_grad_shift[n_ro - 1] = data_grad_shift[n_ro - 2]

    weights = np.sqrt( np.abs(data_grad[:,1]*data_grad_shift[:,1] * data_grad[:,0]*data_grad_shift[:,0]) )
    return weights


def measuredWaveforms(com_meas, grad_meas, seq_params, threshold=0.05):
    """
    Computes the B0 and gradient waveforms from the measurements.
    
    Parameters
    ----------
    com_meas : np.ndarray
        Measurements from center-of-mass measurement sequence.
        Shape: (n_ro_com, n_slices, -/+, n_avgs, x/y/z, n_ch).
    grad_meas : np.ndarray
        Measurements from gradient waveform measurement sequence.
        Shape: (n_ro_grad, n_slices, -/+, n_avgs, x/y/z, n_ch).
    seq_params : dict
        Pulse sequence parameters. Must contain keys: 'dwell_time' (us), 'fov' (mm), 
        'slice_spacing' (mm), 'gamma' (MHz/T).
    threshold : float, optional
        Value between 0 and 1 for thresholding signal and noise from slice-selection. Default: 0.05.
        
    Returns
    -------
    measured_gradient_waveforms_dict : dict
        B0 and gradient waveform measurements for each axis (x/y/z).
    """
    dwell_time = seq_params['dwell_time']
    fov = seq_params['fov']
    slice_spacing = seq_params['slice_spacing']
    gamma = seq_params['gamma']

    data_grad = np.copy(grad_meas)
    data_com = np.copy(com_meas)

    # compute the slice positions (in mm) from the center-of-mass measurements
    data_com = np.moveaxis(data_com, 2, 1)
    slice_positions = centerOfMass(data_com, fov, slice_spacing, threshold=threshold)

    # compute the time evolution of the magnetic fields (in uT)
    data_grad = np.moveaxis(data_grad, 2, 1)
    field_evol = timeEvolField(data_grad, dwell_time, gamma)

    # reshape arrays to prepare for least squares calculation
    slice_positions_flat = np.moveaxis(slice_positions, -2, 0)
    slice_positions_flat = np.reshape(slice_positions_flat, [slice_positions_flat.shape[0], -1])
    del slice_positions

    n_ro_chirp = data_grad.shape[0]
    field_evol_flat = np.moveaxis(field_evol, -2, 0)
    field_evol_flat = np.reshape(field_evol_flat, [field_evol_flat.shape[0], n_ro_chirp, -1])
    del field_evol

    # num of rows of A matrix
    n_samples = slice_positions_flat.shape[1]

    # create weights matrix
    weights = sampleWeights(data_grad)
    weights_flat = np.moveaxis(weights, -2, 0)
    weights_flat = np.reshape(weights_flat, list(weights_flat.shape[0:2]) + [-1])
    del weights

    del data_com, data_grad

    # loop over each axis to compute the measured waveform
    measured_gradient_waveforms = []
    for i_axis in range(3):
        # get weights matrix for this axis
        w_mat = weights_flat[i_axis]
        # create A matrix
        a_mat = np.stack([np.ones(n_samples), slice_positions_flat[i_axis]], axis=1)
        # get field measurements
        y = field_evol_flat[i_axis]
        # weighted least-squares
        x = np.array(
            [np.linalg.lstsq(np.reshape(w_mat[i_ro], [n_samples, 1]) * a_mat, w_mat[i_ro] * y[i_ro], rcond=-1)[0] for i_ro in
             range(n_ro_chirp)]).T
        measured_gradient_waveforms.append(x)
        del w_mat, a_mat, y, x

    return {'x': measured_gradient_waveforms[0],
            'y': measured_gradient_waveforms[1],
            'z': measured_gradient_waveforms[2]}


def computeGirf(waveform_nom, com_meas, grad_meas, seq_params, threshold=0.05, kernel_size=320, lamda=0, max_iter=100):
    """
    Computes the gradient impulse response function from calibration data.
    
    Parameters
    ----------
    waveform_nom : np.ndarray
        Nominal gradient waveform programmed in increments of 10 us.
        Shape: (duration/10 + 1,).
    com_meas : np.ndarray
        Measurements from center-of-mass measurement sequence.
        Shape: (n_ro_com, n_slices, -/+, n_avgs, x/y/z, n_ch).
    grad_meas : np.ndarray
        Measurements from gradient waveform measurement sequence.
        Shape: (n_ro_grad, n_slices, -/+, n_TEs, n_avgs, x/y/z, n_ch).
    seq_params : dict
        Pulse sequence parameters containing keys: 'dwell_time' (us), 'fov' (mm),
        'slice_spacing' (mm), 'gamma' (MHz/T).
    threshold : float, optional
        Value between 0 and 1 for thresholding signal and noise from slice-selection. Default: 0.05.
    kernel_size : int, optional
        Duration of GIRF kernel, increment of 1 us. Default: 320.
    lamda : float, optional
        Regularization parameter for calculating GIRF. Default: 0.
    max_iter : int, optional
        Number of CG iterations for estimating GIRF. Default: 100.
        
    Returns
    -------
    girf_dict : dict
        Gradient impulse response functions (i.e. convolution kernels) for each axis (x/y/z).
    wav_meas : dict
        Gradient waveform measurements for each axis (x/y/z).
    """
    # compute the measured B0 and gradient waveforms
    wav_meas = measuredWaveforms(com_meas, grad_meas, seq_params, threshold=threshold)

    # interpolate the nominal waveform (programmed in 10 us) to 1 us increment
    times_10us = np.arange(0, len(waveform_nom)) * 10
    times_1us = np.arange(0, len(grad_meas))*seq_params['dwell_time']
    wav_nom = np.interp(times_1us, times_10us, waveform_nom)
    n_pts = len(wav_nom)

    # estimate the GIRF for each axis, store in dictionary
    girf_out = {}
    for axis in ['x', 'y', 'z']:
        tmp_w_nom = sp.linop.ConvolveFilter([kernel_size], wav_nom)
        tmp_r = sp.linop.Resize([n_pts], tmp_w_nom.oshape)
        tmp_app = sp.app.LinearLeastSquares(tmp_r * tmp_w_nom, wav_meas[axis][1], lamda=lamda, max_iter=max_iter)
        tmp_ker = tmp_app.run()
        girf_out[axis] = tmp_ker
        del tmp_w_nom, tmp_r, tmp_app, tmp_ker

    return girf_out, wav_meas


def predictedWaveforms(waveform_nom, girf_dict):
    """
    Predicts the true gradient waveforms along each axis using the gradient impulse response function.
    
    Parameters
    ----------
    waveform_nom : np.ndarray
        Nominal gradient waveform programmed in increments of 10 us.
        Shape: (duration/10 + 1,).
    girf_dict : dict
        Gradient impulse response functions (i.e. convolution kernels) for each axis (x/y/z).
        
    Returns
    -------
    waveform_pred : dict
        Predicted waveforms for each axis (x/y/z) in increments of 1 us.
    """
    # interpolate the nominal waveform (programmed in 10 us) to 1 us increment
    times_10us = np.arange(0, len(waveform_nom)) * 10
    times_1us = np.arange(0, 10*(len(waveform_nom)-1)+1)
    waveform_nom_1us = np.interp(times_1us, times_10us, waveform_nom)

    # check if nominal waveform is odd in length
    is_odd = waveform_nom_1us.shape[0]%2 == 1
    # pad with 0 at end to make length even
    if is_odd:
        wav_nom = np.pad(waveform_nom_1us, (0, 1), mode='constant')
    else:
        wav_nom = waveform_nom_1us

    n_ro = wav_nom.shape[0]

    waveform_pred = {}
    for axis in ['x', 'y', 'z']:
        tmp_k = sp.linop.ConvolveData([n_ro], girf_dict[axis])
        tmp_r = sp.linop.Resize([n_ro], tmp_k.oshape)
        tmp_wav_pred = tmp_r * tmp_k * wav_nom

        # take off the padded index if the nominal waveform was odd in length
        if is_odd:
            tmp_wav_pred = tmp_wav_pred[:n_ro-1]

        waveform_pred[axis] = tmp_wav_pred
        del tmp_k, tmp_r, tmp_wav_pred

    return waveform_pred