"""
SequenceTree Arbitrary Gradient Class

Mimics the behavior of SequenceTree's STArbGradient to model 
gradient calculations, k-space trajectory estimation, and Gradient Impulse Response 
Function (GIRF) corrections.

Author: Brian-Tinh Vu
Date: 04/06/2026
Dependencies: numpy, lib.girf.gradient_util, lib.girf.thin_slice_method

Quick Start:
    arb_grad = STArbGradient(ramp_time_1=300, plateau_time=1000, 
                             ramp_time_2=300, readout_duration=1000, 
                             fov=[256, 256, 256], dwell_time=2.0)
    traj = arb_grad.trajectory
"""

import numpy as np
import lib.girf.gradient_util as gu
import lib.girf.thin_slice_method as tsm

# TODO: [MRI Reference] Consider citing literature regarding analytical gradient design or SequenceTree implementation.

class STArbGradient(object):
    """
    Class to model gradient calculation, trajectory estimation, and GIRF correction.
    
    This class may predict the actual gradient waveforms played out by the scanner
    by incorporating system limits and predicting hardware imperfections through
    a user-supplied Gradient Impulse Response Function (GIRF).
    """

    def __init__(
            self,
            ramp_time_1,
            plateau_time,
            ramp_time_2,
            readout_duration,
            fov,
            dwell_time,
            kspace_offset=np.array([0,0,0]),
            gamma=42.5764,
            girf=None,
    ):
        """
        Initializes the STArbGradient.
        
        Parameters
        ----------
        ramp_time_1 : int
            First ramp time of gradient waveform (us); must be a multiple of 10 us.
        plateau_time : int
            Plateau time of the gradient waveform (us); must be a multiple of 10 us.
        ramp_time_2 : int
            Second ramp time of gradient waveform (us); must be a multiple of 10 us.
        readout_duration : int
            Duration of readout ADC coinciding with the gradient. Left-sided. (us)
        fov : list or np.ndarray
            Field-of-view of the acquisition along [x, y, z] in mm.
        dwell_time : float
            ADC dwell time (sampling interval) in us.
        kspace_offset : np.ndarray, optional
            Offset applied to all points in the k-space trajectory along [x, y, z]. Default: [0, 0, 0].
        gamma : float, optional
            Gyromagnetic ratio in MHz/T. Default: 42.5764 (Proton).
        girf : dict, optional
            Gradient impulse response function for predictive correction, 
            sampled on a 1 us increment. Default: None.
        """
        self.ramp_time_1 = ramp_time_1
        self.plateau_time = plateau_time
        self.readout_duration = readout_duration
        self.ramp_time_2 = ramp_time_2
        self.fov = np.array(fov)
        self.dwell_time = dwell_time
        self.kspace_offset = kspace_offset
        self.gamma = gamma
        self.girf = girf

        self._prepare()
        self._peakAmp()
        self._peakSlew()
        self._accumulatedMoment()
        self._trajectory()

    def _prepare(self):
        """
        Computes programmed gradient amplitudes and corresponding gradient zeroth moments.
        
        Calculates gradient waveforms at hardware raster time (often 10 us) and 
        interpolates to a finer raster (1 us) to apply Gradient Impulse Response Function 
        (GIRF) convolution for mitigating eddy currents and system delays.
        """
        # Temporal discretization count of the gradient waveform based on 10 us raster
        N = int(self.duration()/10) + 1

        # Use numerical derivatives to find initial and final amplitudes of the plateau
        initial_amplitude = (self._momentAt(10/self.plateau_time) - self._momentAt(0)) / 10
        final_amplitude = (self._momentAt(1) - self._momentAt(1-10/self.plateau_time)) / 10
        initial_amplitude = np.reshape(initial_amplitude, [3, 1])
        final_amplitude = np.reshape(final_amplitude, [3, 1])

        # Create discrete time arrays to map the theoretical waveform to the discrete amp
        idx = np.array([j for j in range(N)])
        t = np.array([(j*10-self.ramp_time_1)/self.plateau_time for j in range(N)])

        # Map out the 3D gradient amplitudes initialized as zeros
        amp = np.zeros((3, N))
        
        # Populate amplitudes for the first gradient ramp assuming linear slew
        holdt = idx*10/self.ramp_time_1
        holdt = np.stack(3*[holdt], axis=0)
        amp = np.where(t < 0, holdt*initial_amplitude, amp)

        # Set amplitudes across the core gradient plateau by analytically differentiating the moment
        idx_gradWav = np.squeeze(np.argwhere(np.logical_and(t >= 0, t < 1)))
        amp[:, idx_gradWav] = (self._momentAt(t[idx_gradWav]+10/self.plateau_time)-self._momentAt(t[idx_gradWav])) / 10

        # Populate amplitudes for the second gradient ramp assuming linear slew
        holdt = 1 - (idx*10-self.ramp_time_1-self.plateau_time)/self.ramp_time_2
        holdt = np.stack(3 * [holdt], axis=0)
        amp = np.where(t >= 1, holdt*final_amplitude, amp)
        
        # Nominally programmed gradient at 10 us resolution
        self.amp = amp

        # Interpolate programmed shape from 10 us onto a finer 1 us grid to support precise GIRF convolution
        times_10us = t*self.plateau_time
        times_1us = np.linspace(times_10us[0], times_10us[len(times_10us)-1], 10*(len(times_10us)-1)+1)
        amp_1us = np.zeros((3, len(times_1us)))
        for i_axis in range(3):
            amp_1us[i_axis] = np.interp(times_1us, times_10us, self.amp[i_axis])
        self.amp_1us = amp_1us


        # GIRF correction inspired by the following references. However, the present implementation opted to estimate
        # the GIRF and effective gradient waveform by deconvolution and convolution, respectively.

        # Vannesjo, S. J., Haeberlin, M., Kasper, L., Pavan, M., Wilm, B. J., Barmet, C., & Pruessmann, K. P. (2013). 
        # Gradient system characterization by impulse response measurements with a dynamic field camera. 
        # Magnetic Resonance in Medicine, 69(2), 583–593. https://doi.org/10.1002/mrm.24263

        # Addy, N. O., Wu, H. H., & Nishimura, D. G. (2012). 
        # Simple method for MR gradient system characterization and k-space trajectory estimation. 
        # Magnetic Resonance in Medicine, 68(1), 120–129. https://doi.org/10.1002/mrm.23217
        # Convolve physical GIRF predictions on the 1 us waveform to capture hardware distortions
        if self.girf is not None:
            self.amp_1us = self._correctAmp(self.girf, self.amp_1us)


        # Integrate the gradient array to yield cumulative k-space gradient moments
        total_moment = self.amp_1us * 1
        total_moment = np.cumsum(total_moment, axis=1)
        self.total_moment = total_moment

    def _peakAmp(self):
        """Finds maximum amplitude across all gradient axes."""
        self.peakamp = np.max(np.abs(self.amp))

    def _peakSlew(self):
        """Approximates maximum slew rate across all gradient axes using numerical differences."""
        N = int(self.duration() / 10) + 1
        tmp_amp_tp10 = self.amp[:, 1:]
        tmp_amp_t = self.amp[:, :N-1]
        slew = (tmp_amp_tp10 - tmp_amp_t)/10
        self.peakslew = np.max(np.abs(slew))

    def _accumulatedMoment(self):
        """
        Computes the accumulated zeroth moment factoring in the encoding sequence.
        
        Assumes the pre-phasing encoding gradient fully nulls the moment acquired during
        the first gradient ramp, setting up the exact starting location in k-space.
        """
        # Determine the encoding moment needed to null ramp 1
        tmp_encode_moment = -self._ramp1Moment() + self._momentAt(0)
        # Trajectory evolution is the initial spatial encoding plus the ongoing accumulation
        self.accumulated_moment = (tmp_encode_moment.T + self.total_moment.T).T

    def _trajectory(self):
        """
        Computes the k-space trajectory traversed during the ADC window.
        
        Sub-samples the continuous accumulated gradient moment at intervals 
        corresponding to the ADC dwell time.
        """
        # Extract timings for gradient window
        times = np.arange(0, self.duration()+1) - self.ramp_time_1

        # Isolate readout indices occurring during the ADC ON window
        tmp_cond = np.logical_and(times >= 0, times < self.readout_duration)

        idx_roGrad = np.squeeze(np.argwhere(tmp_cond))
        tmp_roGrad_accum_mom = self.accumulated_moment[:, idx_roGrad]
        
        # Subsample high temporal resolution moment onto the explicit ADC sampling grid
        tmp_roGrad_accum_mom = tmp_roGrad_accum_mom[:, ::self.dwell_time]
        self.trajectory = gu.moment2kspace(tmp_roGrad_accum_mom, self.fov, gamma=self.gamma)

    def duration(self):
        """
        Calculates total duration of the arbitrary gradient pulse block.
        
        Returns
        -------
        duration : int
            Total physical time of STArbGradient spanning ramp up, plateau, and ramp down (us).
        """
        return self.ramp_time_1 + self.plateau_time + self.ramp_time_2

    def _momentAt(self, t):
        """
        Returns spatial gradient moments corresponding to a non-dimensional timeframe.
        
        Parameters
        ----------
        t : float or np.ndarray
            Parameterization mapping the gradient timeframe as 0 <= t <= 1.
            
        Returns
        -------
        moments : ndarray
            Gradient spatial moment at relative time t along Cartesian axes [x, y, z].
        """
        tmp_kspace_offset = self.kspace_offset
        if not (isinstance(t, float) or isinstance(t, int)):
            tmp_kspace_offset = np.reshape(tmp_kspace_offset, [3] + len(t.shape)*[1])
        return gu.kspace2moment(self._gradientShape(t) + tmp_kspace_offset, self.fov, gamma=self.gamma)

    def _gradientShape(self, t):
        """
        Prototype method to output desired k-space shape across time. To be overridden by subclasses.
        
        Parameters
        ----------
        t : float or np.ndarray
            Parameterization of the gradient from 0 to 1.
            
        Returns
        -------
        kspace : ndarray or None
            Target sequence framework shape to follow.
        """
        return

    def _ramp1Moment(self):
        """Calculates trapezoidal moment contribution exclusively from the first ramp."""
        initial_amplitude = (self._momentAt(10/self.plateau_time) - self._momentAt(0)) / 10
        return initial_amplitude * self.ramp_time_1 / 2

    def _correctAmp(self, girf, amp):
        """
        Applies system measured GIRFs to predict actual realized gradient outputs.
        
        The gradient impulse response function (GIRF), measured through the thin-slice algorithm,
        contains measured characterizations of system delays and eddy current dynamics. The effective gradient 
        waveform is estimated by a convolution of the programmed gradient waveform with the GIRF kernel.
        
        Parameters
        ----------
        girf : dict
            GIRF function parameters.
        amp : np.ndarray
            Ideally programmed gradient amplitudes. shape: (3, n_samples)
            
        Returns
        -------
        newamp : np.ndarray
            Realistic gradient amplitudes distorted by system imperfections.
        """
        newamp = np.copy(amp)
        # Compute true gradient output by convolving against each physical axis impulse response
        # TODO: implement a rotation of the FOV before GIRF correction
        axes = ['x', 'y', 'z']
        for i_axis, axis in enumerate(axes):
            newamp[i_axis] = tsm.predictedWaveforms(self.amp[i_axis], girf)[axis]
        return newamp

    def _rounduptime(self, t):
        """Ceiling division mapping durations to the 10us system clock raster."""
        return int(t/10 + 0.999999) * 10


# TODO: [MRI Reference] Add citation here for sequence techniques implementing circular EPI / Rosette.
class STCircleGradient(STArbGradient):
    """
    STArbGradient subtype simulating SequenceTree's circular or complex gradient trajectories.
    """

    def __init__(
            self,
            kspace_radius_1,
            kspace_radius_2,
            num_cycles,
            kspace_direction_1,
            kspace_direction_2,
            ramp_time_1,
            plateau_time,
            ramp_time_2,
            fov,
            dwell_time,
            kspace_offset=np.array([0, 0, 0]),
            gamma=42.5764,
    ):
        """
        Initializes an STCircleGradient instance.
        
        Parameters
        ----------
        kspace_radius_1 : float
            Primary circular radius target in k-space.
        kspace_radius_2 : float
            Secondary perpendicular radius target in k-space.
        num_cycles : int
            Rotational cycles traversed during reading duration.
        kspace_direction_1 : np.ndarray
            Primary geometric orientation vector array [x,y,z].
        kspace_direction_2 : np.ndarray
            Secondary geometric orientation vector array [x,y,z].
        ramp_time_1 : int
            First ramp time in us.
        plateau_time : int
            Plateau/readout duration in us.
        ramp_time_2 : int
            Second ramp time in us.
        fov : list or np.ndarray
            Field-of-view in mm.
        dwell_time : float
            ADC dwell time step.
        kspace_offset : np.ndarray, optional
            A translation coordinate in k-space [x,y,z]. Default: [0,0,0].
        gamma : float, optional
            Gyromagnetic ratio. Default: 42.5764.
        """
        self.kspace_radius_1 = kspace_radius_1
        self.kspace_radius_2 = kspace_radius_2
        self.num_cycles = num_cycles
        self.kspace_direction_1 = kspace_direction_1
        self.kspace_direction_2 = kspace_direction_2
        super().__init__(ramp_time_1, plateau_time, ramp_time_2, fov, dwell_time, kspace_offset=kspace_offset, gamma=gamma)


    def _gradientShape(self, t):
        """
        Defines the dynamic parametric trajectory forming mathematical circles or sinusoids.
        
        Parameters
        ----------
        t : float or np.ndarray
            Parameterization timeframe (0 to 1).
            
        Returns
        -------
        kspace : np.ndarray
            Circular Cartesian mapping vector at relative time t.
        """
        t2 = t * 2 * 3.141592*self.num_cycles
        tmp_kspace_direction_1 = self.kspace_direction_1
        tmp_kspace_direction_2 = self.kspace_direction_2
        if not (isinstance(t, float) or isinstance(t, int)):
            tmp_kspace_direction_1 = np.reshape(tmp_kspace_direction_1, [3] + len(t.shape)*[1])
            tmp_kspace_direction_2 = np.reshape(tmp_kspace_direction_2, [3] + len(t.shape)*[1])
            
        # Modulate directional bases with sine and cosine to construct the geometric circle
        tmp_x = tmp_kspace_direction_1*self.kspace_radius_1*np.cos(t2)
        tmp_y = tmp_kspace_direction_2*self.kspace_radius_2*np.sin(t2)
        return tmp_x + tmp_y