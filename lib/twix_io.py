"""
Siemens TWIX Data I/O for MRI

Read Siemens TWIX .dat files for MRI raw k-space data processing.
Supports XA format with Cartesian undersampling masks.

Author: Brian-Tinh Vu
Date: 11/26/2025
Dependencies: numpy

Quick Start:
    data = readSiemensXA('measurement.dat', nviews=128)
    ksp = readFromCartesianMask('measurement.dat', mask, n_prep=100)
"""
import numpy as np
from lib.st_interface import getPeIdxsFromMask
import twixtools

def readSiemensXA(datfile, nviews):
    """
    Read Siemens TWIX .dat file (XA format) for k-space data.
    
    Parses binary TWIX file format, skips PMUDATA blocks, and extracts
    multi-channel k-space data with progress reporting.
    
    Parameters
    ----------
    datfile : str
        Path to Siemens TWIX .dat file (XA format).
    nviews : int
        Number of phase encoding lines to read.
    
    Returns
    -------
    data : np.ndarray
        Complex k-space data shape: (num_readouts, nviews, num_channels).
    
    Notes
    -----
    Skips PMUDATA blocks and extracts imaging data from XA30 format.
    Handles interleaved real/imaginary data conversion.
    
    TODO: Add safety check for additional ADCs acquired but not read.
    """
    with open(datfile, "rb") as f:
        # Skip header and calculate measurement length
        _ = np.fromfile(f, dtype=np.uint32, count=1)
        n_meas = int(np.fromfile(f, dtype=np.uint32, count=1)[0])

        meas_len = 0

        # Calculate total measurement length from all measurements except last
        for _ in range(n_meas - 1):
            f.seek(16, 1)
            tmp = int(np.fromfile(f, dtype=np.uint64, count=1)[0])
            meas_len = meas_len + int(np.ceil(tmp / 512.0) * 512)
            f.seek(152 - 24, 1)

        # Seek to data section and read header
        offset = 2 * 4 + 152 * 64 + 126 * 4 + meas_len
        f.seek(offset, 0)
        header_size = int(np.fromfile(f, dtype=np.uint32, count=1)[0])
        _ = np.fromfile(f, dtype=np.uint8, count=header_size - 4)

        # Skip PMUDATA blocks and find first imaging data
        num_readouts = 0
        cnt = 0

        while num_readouts == 0:
            _ = np.fromfile(f, dtype=np.uint8, count=48)
            nr_arr = np.fromfile(f, dtype=np.uint16, count=1)
            if nr_arr.size == 0:
                raise EOFError("Unexpected end of file while searching for first readout count.")
            num_readouts = int(nr_arr[0])

            if num_readouts == 0:
                # Skip PMUDATA block
                f.seek(-50, 1)
                ul_dma_length = 184
                mdh_start = -ul_dma_length
                data_u8 = np.fromfile(f, dtype=np.uint8, count=ul_dma_length)
                data_u8 = data_u8[mdh_start:]
                data_u8[3] = (data_u8[3] & 1)
                ul_dma_length = int(np.frombuffer(data_u8[0:4].tobytes(), dtype=np.uint32)[0])
                _ = np.fromfile(f, dtype=np.uint8, count=ul_dma_length - 184)
                cnt += 1

        # Get number of channels and prepare data array
        num_channels = int(np.fromfile(f, dtype=np.uint16, count=1)[0])
        f.seek(-(48 + 4), 1)

        print(f"Reading: {datfile}\nProgress...")
        data = np.zeros((num_readouts, nviews, num_channels), dtype=np.complex64)

        # Read k-space data for all phase encoding lines
        i_pe = 0
        percent_finished = 0

        while True:
            i_pe += 1

            # Read header for current phase encoding line
            _ = np.fromfile(f, dtype=np.uint8, count=48)
            nr_arr = np.fromfile(f, dtype=np.uint16, count=1)
            if nr_arr.size == 0:
                raise EOFError("Unexpected end of file while reading readout count.")
            num_readouts_now = int(nr_arr[0])

            if num_readouts_now == 0:
                # Skip PMUDATA block
                f.seek(-50, 1)
                ul_dma_length = 184
                mdh_start = -ul_dma_length
                data_u8 = np.fromfile(f, dtype=np.uint8, count=ul_dma_length)
                data_u8 = data_u8[mdh_start:]
                data_u8[3] = (data_u8[3] & 1)
                ul_dma_length = int(np.frombuffer(data_u8[0:4].tobytes(), dtype=np.uint32)[0])
                _ = np.fromfile(f, dtype=np.uint8, count=ul_dma_length - 184)
                i_pe -= 1
                continue

            # Skip remaining header bytes
            _ = np.fromfile(f, dtype=np.uint8, count=192 - 50)

            # Read data for each channel
            for i_ch in range(num_channels):
                _ = np.fromfile(f, dtype=np.uint8, count=32)
                tmp = np.fromfile(f, dtype=np.float32, count=num_readouts_now * 2)
                
                # Convert interleaved real/imag to complex
                real_part = tmp[0::2]
                imag_part = tmp[1::2]
                tmp_complex = real_part + 1j * imag_part
                
                data[:num_readouts_now, i_pe - 1, i_ch] = tmp_complex

            # Update progress indicator
            new_percent = int((100 * i_pe) / nviews)
            if new_percent > percent_finished + 9:
                percent_finished = new_percent
                print(f"{percent_finished:3d} %")

            if i_pe == nviews:
                break

    return data


def readFromCartesianMask(datfile, mask, n_prep, mode='2D'):
    """
    Read TWIX data using Cartesian undersampling mask.
    
    Parameters
    ----------
    datfile : str
        Path to Siemens TWIX .dat file.
    mask : np.ndarray
        Binary sampling mask (1=sampled, 0=skipped).
        Shape: (Nx, Ny) for 2D, (Ny, Nz) for 3D.
    n_prep : int
        Number of preparation lines to skip.
    mode : {'2D', '3D'}, optional
        Acquisition mode. Default: '2D'.
    
    Returns
    -------
    ksp : np.ndarray
        K-space data with channel-first ordering.
        Shape: (num_channels, Nx, Ny) for 2D,
        (num_channels, Nx, Ny, Nz) for 3D.
    
    Raises
    ------
    ValueError
        If mode is not '2D' or '3D'.
    
    Notes
    -----
    For 3D Cartesian, assumes fast PE dimension is along Ny.
    Handles SequenceTree to Python index conversion.
    """
    if mode == '2D':
        pe_idxs = getPeIdxsFromMask(mask, mode=mode)

        # Read raw data from TWIX
        n_lines_acquired = len(pe_idxs) + n_prep

        data = readSiemensXA(datfile, n_lines_acquired)
        data = data[:,n_prep:,:]

        n_ro = data.shape[0]
        n_ch = data.shape[-1]
        n_y = mask.shape[1]

        # Initialize complex k-space matrix
        ksp = np.zeros((n_ro, n_y, n_ch), dtype=np.complex64)

        # Place data in proper k-space locations
        ksp[:, pe_idxs, :] = data

        ksp = np.moveaxis(ksp, -1, 0)

        return ksp

    elif mode == '3D':
        pe_idxs_flat = getPeIdxsFromMask(mask, mode=mode)

        n_y = mask.shape[0]
        n_z = mask.shape[1]
        
        # Get PE indices from SequenceTree (-N/2 to N/2 convention)
        pe1 = np.mod(pe_idxs_flat, n_y) - int(n_y/2)
        pe2 = pe_idxs_flat//n_y - int(n_z/2)

        # Shift PE indices to Python mask coordinates
        pe1 = pe1 + int(n_y/2)
        pe2 = pe2 + int(n_z/2)

        # Read raw data from TWIX
        n_lines_acquired = np.count_nonzero(mask) + n_prep
        data = readSiemensXA(datfile, n_lines_acquired)
        data = data[:,n_prep:,:]

        n_ro = data.shape[0]
        n_ch = data.shape[-1]
        # Initialize complex k-space matrix
        ksp = np.zeros((n_ro, n_y, n_z, n_ch), dtype=np.complex64)

        # Place data in proper k-space locations
        ksp[:, pe1, pe2, :] = data  

        ksp = np.moveaxis(ksp, -1, 0)

        return ksp

    else:
        raise ValueError(f"Invalid mode: {mode}. Must be '2D' or '3D'.")


def getFOVTransformation(datfile):
    """
    Reads in a Siemens .twix raw data file and gets the FOV transform for that acquisition.
    
    Parameters
    ----------
    datfile : str
        Path to the Siemens TWIX .dat file.
        
    Returns
    -------
    dict
        Dictionary containing 'normal', 'offset', and 'inplane_rot' transformation parameters.
    """
    twix_file = twixtools.read_twix(datfile, parse_prot=True, parse_data=True, parse_pmu=False, parse_geometry=True)

    # get normal, offset, and rotation parameters of acquisition
    normal = np.array(twix_file[-1]['geometry'][0].normal)
    offset = np.array(twix_file[-1]['geometry'][0].offset)
    inplane_rot = twix_file[-1]['geometry'][0].inplane_rot

    return {
        "normal": normal,
        "offset": offset,
        "inplane_rot": inplane_rot 
    }