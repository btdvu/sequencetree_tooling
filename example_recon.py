# Example of image reconstrution with SigPy.
# Intended to be used to test image reconstruction with HPC.

import numpy as np
import sigpy as sp
import sigpy.mri as mr
import sigpy.plot as pl

import lib
from lib.twix_io import read_twix_siemens_XA
from lib.radial import trajGoldenAngle, trajCalibration, estGradDelayMultiCh, shiftTrajectory


if __name__ == '__main__':
    device = sp.cpu_device
    
    # read in raw data
    filename = 'data/meas_MID00292_FID53883_st_spgr_stackOfStars_20251208_256sp.dat'
    n_calibPairs = 64
    n_spokesPerSlice = 256
    n_slice = 128
    n_views = n_calibPairs*2 + n_spokesPerSlice*n_slice
    data = read_twix_siemens_XA(filename, n_views)
    data = np.moveaxis(data, (0, 1, 2), (2, 1, 0))
    n_ro = data.shape[2]
    n_coil = data.shape[0]

    # sort calibration data and image acquisition data
    ksp_calib = data[:,:2*n_calibPairs]
    ksp = data[:,2*n_calibPairs:]
    ksp = np.reshape(ksp, (n_coil, n_slice, n_spokesPerSlice, n_ro))

    # get the nominal trajectories
    coord_calib_nom = trajCalibration(n_ro, n_calibPairs)
    coord_nom = trajGoldenAngle(n_ro, n_spokesPerSlice)

    # use the calibration data to estimate the trajectory shift matrix
    shift_matrix = estGradDelayMultiCh(ksp_calib, coord_calib_nom)
    # apply the shift matrix to the image acquisition data
    coord = shiftTrajectory(coord_nom, shift_matrix)

    # perform an IFFT along the z-axis
    ksp_2d = sp.ifft(ksp, axes=(1,))

    # select a single 2d slice from the 3D stack-of-stars acquisition
    i_slice = n_slice//2
    ksp_slice = ksp_2d[:,i_slice]

    print("PAUSE")

    # estimate coil sensitivities
    mps = mr.app.JsenseRecon(ksp_slice, lamda=1E-3, device=device, coord=coord, img_shape=2*[n_ro]).run()

    # perform Sense Recon
    img = mr.app.SenseRecon(ksp_slice, mps, lamda=1E-3, coord=coord, device=device, max_iter=30).run()


    print("STOP")
