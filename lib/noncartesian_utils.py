"""
Non-Cartesian Utilities

Utility functions for non-Cartesian trajectories.

Author: Brian-Tinh Vu
Date: 02/27/2026
Dependencies: numpy, sigpy

Quick Start:
    shift = [5, 5]
    ksp_adjusted = shiftFieldOfView(ksp, coord, shift)
"""

import numpy as np
import sigpy as sp


# TODO: Rigorous testing of shifts and rotations with SequenceTree sequences.
# TODO: Also test multiple non-Cartesian trajectories with non-constant readout gradient amplitude sampling. (e.g. spiral, ramp-sampled radial, EPI)
# TODO: Determine whether FOV shifting needs to be done before or after gradient delay estimation. I think before, but am not sure until you try.
def shiftFieldOfView(ksp, coord, shift, img_shape=None):
    """
    Compute phase adjustment to the k-space data based on the coordinates and the computed image offset (in pixels).
    
    Parameters
    ----------
    ksp : np.ndarray
        K-space data.
    coord : np.ndarray
        Coordinates corresponding to the k-space data.
    shift : array_like
        Desired shift in pixels.
    img_shape : tuple or list, optional
        Shape of the image. Default: None (estimated from coord).
        
    Returns
    -------
    ksp_adjusted : np.ndarray
        Phase-adjusted k-space data.
    """
    if img_shape is None:
        img_shape = sp.estimate_shape(coord)
    elif coord.shape[-1] != len(img_shape):
        raise ValueError(f"Coordinate matrix dimensions ({coord.shape[-1]}-dim) do not match image shape ({len(img_shape)}-dim)")
    elif coord.shape[-1] != len(shift):
        raise ValueError(f"Coordinate matrix dimensions ({coord.shape[-1]}-dim) do not match shift dimensions ({len(shift)}-dim)")

    # NOTE: x-dim phase adjustment needs to be multiplied by a factor of -1, based on Siemens coordinate system; found empirically
    modified_shift = np.zeros_like(shift)
    modified_shift[0] = -1*shift[0]
    modified_shift[1] = 1*shift[1]
    if len(shift) > 2:
        modified_shift[2] = 1*shift[2]

    phase_adjustment_per_direction = coord/img_shape * 2*np.pi * modified_shift
    
    phase_adjustment = np.sum(phase_adjustment_per_direction, axis=-1)

    adjustment = np.exp(-1j*phase_adjustment)

    ksp_adjusted = ksp*adjustment

    return ksp_adjusted    


def offsetInImagingPlane(transform, voxel_size=None):
    """
    Gets the offset vector in the imaging plane, which can be used to adjust phase of k-space data to perform an FOV shift.
    
    If 'voxel_size' is provided, convert the offset (which is expressed in mm) to number of pixels.
    Most appropriate to convert from physical units (e.g. mm) to number of pixels when the offset vector is expressed in the imaging plane coordinates.
    
    Parameters
    ----------
    transform : dict
        FOV transform parameters provided by the scanner (expected keys: 'normal', 'offset', 'inplane_rot').
    voxel_size : array_like, optional
        Size of the voxels. If provided, converts the offset in mm to number of pixels. Default: None.
        
    Returns
    -------
    np.ndarray
        Offset vector in the imaging plane (either in mm or in pixels).
    """
    offset = transform['offset']

    # Get the axes of the imaging plane, expressed in the coordinate system of the lab frame.
    imaging_plane_axes = _imagingPlaneAxes(transform)

    # Offset in imaging plane coordinates.
    offset_in_imaging_plane_coord = _transformToImagingPlaneAxes(offset, imaging_plane_axes)

    if voxel_size is not None:
        offset_in_imaging_plane_coord_in_pixels = offset_in_imaging_plane_coord/np.array(voxel_size)
        return offset_in_imaging_plane_coord_in_pixels
    
    return offset_in_imaging_plane_coord


def _imagingPlaneAxes(transform):
    """
    Get the unit vectors of the coordinate system in the imaging plane (x', y', and z').
    
    Parameters
    ----------
    transform : dict
        FOV transform parameters provided by the scanner.
        
    Returns
    -------
    dict
        Dictionary containing 'x_prime', 'y_prime', and 'z_prime' unit vectors.
    """
    normal = transform['normal']
    offset = transform['offset']
    inplane_rot = transform['inplane_rot']

    # Determine the anatomical plane that our FOV rotation is derived from.
    # TODO: We think it is determined by the most dominant vector component in our normal vector.
    # For example, if the largest component in 'normal' is in the z direction, the FOV rotation is derived from the axial plane.
    # If the largest component in 'normal' is in y, the FOV rotation is derived from the coronal plane.
    # And if the largest component in 'normal' is in x, the FOV rotation is derived from the sagittal plane.
    # Need to validate this with more 2D experiments.
    # Set the lab frame x, y, and z directions based on the index of the largest vector component in 'normal'.
    idx_largest_component = np.argmax(np.abs(normal))

    # Anatomical frame (axial, sagittal, coronal) unit vectors.
    if idx_largest_component == 2:
        z = np.array([0,0,1])
        x = np.array([1,0,0])
        y = np.array([0,1,0])
    elif idx_largest_component == 1:
        z = np.array([0,1,0])
        x = np.array([0,0,1])
        y = np.array([1,0,0])
    else:
        z = np.array([1,0,0])
        x = np.array([0,1,0])
        y = np.array([0,0,1])

    # z' is simply the unit normal vector.
    z_prime = normal/np.linalg.norm(normal)

    # Get the angle of rotation and rotation axis between starting and new normal vectors.
    angle_of_rotation = np.arccos(np.dot(z, z_prime))
    rotation_axis = np.cross(z, z_prime)
    rotation_axis = rotation_axis/np.linalg.norm(rotation_axis)

    # Get the rotation matrix that rotates the starting normal to the new normal.
    rot_matrix = _rotationFromAxisAndAngle(rotation_axis, angle_of_rotation)
    tmp_x_prime = rot_matrix @ x
    tmp_y_prime = rot_matrix @ y

    # Apply the in-plane rotation.
    in_plane_rot_matrix = _rotationFromAxisAndAngle(z_prime, -inplane_rot)
    x_prime = in_plane_rot_matrix @ tmp_x_prime
    y_prime = in_plane_rot_matrix @ tmp_y_prime

    return {
        "x_prime": x_prime,
        "y_prime": y_prime,
        "z_prime": z_prime 
    }


def _transformToImagingPlaneAxes(vector, imaging_plane_axes):
    """
    Recalculates a vector's components, expressed in the lab frame coordinates, based on the coordinate system of the imaging plane.
    
    Parameters
    ----------
    vector : np.ndarray
        Vector in lab frame coordinates.
    imaging_plane_axes : dict
        Dictionary containing 'x_prime', 'y_prime', and 'z_prime' unit vectors.
        
    Returns
    -------
    np.ndarray
        Vector in the imaging plane coordinate system.
    """
    x_prime = imaging_plane_axes['x_prime']
    y_prime = imaging_plane_axes['y_prime']
    z_prime = imaging_plane_axes['z_prime']

    # Create the matrix which transforms a vector in the lab frame to one in the imaging frame.
    transform_lab_to_imaging_plane = np.stack([x_prime, y_prime, z_prime])

    return transform_lab_to_imaging_plane @ vector


def _rotationFromAxisAndAngle(axis, angle):
    """
    Given a rotation axis and the angle of rotation, return the rotation matrix.
    
    Parameters
    ----------
    axis : np.ndarray
        Axis of rotation.
    angle : float
        Angle of rotation in radians.
        
    Returns
    -------
    np.ndarray
        3x3 rotation matrix.
    """
    # Got this from Wikipedia (https://en.wikipedia.org/wiki/Rotation_matrix).
    r_11 = axis[0]**2*(1-np.cos(angle)) + np.cos(angle)
    r_12 = axis[0]*axis[1]*(1-np.cos(angle)) - axis[2]*np.sin(angle)
    r_13 = axis[0]*axis[2]*(1-np.cos(angle)) + axis[1]*np.sin(angle)

    r_21 = axis[0]*axis[1]*(1-np.cos(angle)) + axis[2]*np.sin(angle)
    r_22 = axis[1]**2*(1-np.cos(angle)) + np.cos(angle)
    r_23 = axis[1]*axis[2]*(1-np.cos(angle)) - axis[0]*np.sin(angle)

    r_31 = axis[0]*axis[2]*(1-np.cos(angle)) - axis[1]*np.sin(angle)
    r_32 = axis[1]*axis[2]*(1-np.cos(angle)) + axis[0]*np.sin(angle)
    r_33 = axis[2]**2*(1-np.cos(angle)) + np.cos(angle)

    return np.array([   [r_11, r_12, r_13],
                        [r_21, r_22, r_23],
                        [r_31, r_32, r_33]])