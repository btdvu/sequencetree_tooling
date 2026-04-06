"""
SequenceTree Archimedean Spiral Gradient Class

Mimics the behavior of SequenceTree's spiral readouts, mapping 
an Archimedean spiral parametric trajectory to physical gradient waveforms 
with optional GIRF distortion prediction.

Analytic expressions for Archimedean spirals given by:

Bernstein, M. A., King, K. F., & Zhou, X. J. (2004). Handbook of MRI Pulse Sequences. 
San Diego, CA: Elsevier Academic Press. (see Section 17.6).

Glover, G. H. (1999). Simple analytic spiral K-space algorithm. Magnetic Resonance in Medicine, 42(2), 412–415. 
https://doi.org/10.1002/(SICI)1522-2594(199908)42:2<412::AID-MRM25>3.0.CO;2-U

Author: Brian-Tinh Vu
Date: 04/06/2026
Dependencies: numpy, lib.girf.STArbGradient

Quick Start:
    traj = traj(n_shots=16, matrix_size=256, maxamp=40.0, ramprate=150.0, 
                ramp_time_1=300, ramp_time_2=300, fov_xy=220.0, dwell_time=2.0)
"""

import numpy as np
from lib.girf.STArbGradient import STArbGradient

# TODO: [MRI Reference] Consider formalizing the citation to "Bernstein, M. A., King, K. F., & Zhou, X. J. Handbook of MRI pulse sequences. Elsevier, 2004 (Section 17.6)" and related foundational papers (e.g., Glover 1999) on spiral k-space trajectory design.

def traj(n_shots, matrix_size, maxamp, ramprate, ramp_time_1, ramp_time_2, fov_xy, dwell_time, gamma=42.5764, girf=None):
    """
    Generates a multi-shot Archimedean spiral k-space trajectory.
    
    Analytically designs the gradient waveforms necessary to traverse an Archimedean
    spiral within hardware limits (maximum amplitude and slew rate), and 
    computes the resulting k-space path. Incorporates GIRF if provided.
    
    Parameters
    ----------
    n_shots : int
        Number of spiral shots (interleaves) required to fully sample k-space.
    matrix_size : int
        Number of points forming the targeted image grid across the diameter.
    maxamp : float
        Maximum gradient amplitude limit (uT/mm).
    ramprate : float
        Maximum gradient slew rate limit (uT/mm/us).
    ramp_time_1 : int
        First ramp time (us).
    ramp_time_2 : int
        Second ramp time (us).
    fov_xy : float
        Field-of-view spanning the x and y directions (mm).
    dwell_time : float
        ADC sampling interval (us).
    gamma : float, optional
        Gyromagnetic ratio in MHz/T. Default: 42.5764 (Proton).
    girf : dict, optional
        Gradient impulse response function for predictive correction, 
        sampled on a 1 us increment. Default: None.
        
    Returns
    -------
    trajectory : np.ndarray
        Array containing physical k-space coordinates for the complete spiral acquisition.
        shape: (n_shots, n_samples_per_shot, 2)
    """
    # Calculate geometric rotation offsets for each interleaved shot
    shot_angles = [2*np.pi/n_shots*i for i in range(n_shots)] # rad
    
    # Generate spatial gradient waveform models for each spiral leaf
    spiral_gradients = []
    for shot_angle in shot_angles:
        tmp_grad = STSpiralGradient(shot_angle, n_shots, matrix_size, maxamp, ramprate, ramp_time_1, ramp_time_2, fov_xy, 
                                    dwell_time, gamma=gamma, girf=girf)
        spiral_gradients.append(tmp_grad)
        del tmp_grad

    # Horizontally aggregate the 2D Cartesian k-space points output from each shot model
    trajectory = np.stack([spiral_gradients[i].trajectory for i in range(len(spiral_gradients))], axis=-1).T
    trajectory = trajectory[:, :, 0:2]

    return trajectory


# TODO: use kwargs to clean up this class's argument list
class STSpiralGradient(STArbGradient):
    """
    SequenceTree-aware implementation of an Archimedean spiral gradient.
    
    Each instance physically models the waveform dynamics of a single spiral shot
    (readout) originating from the k-space center. Inherits the structural GIRF
    and moment discretization behaviors of STArbGradient.
    """

    def __init__(
            self,
            shot_angle,
            n_shots,
            matrix_size,
            maxamp,
            ramprate,
            ramp_time_1,
            ramp_time_2,
            fov_xy,
            dwell_time,
            kspace_offset=np.array([0, 0, 0]),
            gamma=42.5764,
            girf=None,
    ):
        """
        Initializes an STSpiralGradient instance for a single shot.
        
        Parameters
        ----------
        shot_angle : float
            Starting rotational angle of this specific spiral shot (rad).
        n_shots : int
            Total number of interleaved spiral shots.
        matrix_size : int
            Targeted reconstructed image width/height in voxels.
        maxamp : float
            Maximum gradient amplitude threshold (uT/mm).
        ramprate : float
            Maximum gradient slew rate threshold (uT/mm/us).
        ramp_time_1 : int
            First ramp time in us.
        ramp_time_2 : int
            Second ramp time in us.
        fov_xy : float
            Isotropic Field-of-view dimension in mm.
        dwell_time : float
            Time elapsed between ADC samples (us).
        kspace_offset : np.ndarray, optional
            Translational shift added to k-space coordinates across all axes [x,y,z]. Default: [0, 0, 0].
        gamma : float, optional
            Gyromagnetic ratio. Default: 42.5764.
        girf : dict, optional
            GIRF convolution dictionary for system hardware predictions. Default: None.
        """
        self.shot_angle = shot_angle
        self.n_shots = n_shots
        self.matrix_size = matrix_size
        self.maxamp = maxamp
        self.ramprate = ramprate
        self.fov_xy = fov_xy
        fov = 2*[fov_xy] + [1] # z dim is a dummy parameter, spiral is 2D
        self.gamma = gamma
        self.dwell_time = dwell_time

        # Calculate physically limited angular and temporal transition points
        self._intermediateParams()

        super().__init__(ramp_time_1, self.plateau_time, ramp_time_2, self.readout_duration, fov, dwell_time,
                         kspace_offset=kspace_offset, gamma=gamma, girf=girf)

    def _intermediateParams(self):
        """
        Computes analytical constants governing the spiral's parametric equations.
        
        Evaluates hardware boundaries to divide the k-space trajectory into 
        slew-rate-limited and amplitude-limited kinetic phases, dictating
        how quickly k-space radial distance (theta) progresses.
        """
        lamda = self.n_shots/(2*np.pi*self.fov_xy/10)
        beta = self.gamma*1E2 * self.ramprate*1E5 / lamda
        a2 = (9*beta/4) ** (1/3)
        # Ts is the transition time between slew-rate limited segment and the gradient amp limited segment
        Ts = ((3 * self.gamma*1E2 * self.maxamp/10) / (2*lamda*a2**2))**3
        # theta_s is the transition angle between the slew-rate limited segment and the gradient amp limited segment
        theta_s = (0.5 * beta * Ts**2) / (1 + beta/(2*a2) * Ts**(4/3))
        # theta_max defines the outer radial perimeter of the targeted k-space diameter
        theta_max = np.pi*self.matrix_size/self.n_shots

        # Set physical phase boundaries
        self.lamda = lamda
        self.beta = beta
        self.a2 = a2
        self.Ts = Ts
        self.theta_s = theta_s
        self.theta_max = theta_max

        # Determine if gradient achieves amplitude limits before resolving the required outer FOV limits
        if theta_s < theta_max:
            self.isGradAmpLimited = True
            Tacq = Ts + lamda / (2 * self.gamma*1E2 * self.maxamp/10) * (theta_max**2 - theta_s**2)
        else:
            self.isGradAmpLimited = False
            Tacq = (2*np.pi * self.fov/10) / (3 * self.n_shots) \
                   * (2 * self.gamma*1E2 * self.ramprate*1E5 * (self.fov/10/self.matrix_size)**3)**(-1/2)
        self.Tacq = Tacq

        # Calculate exact acquisition duration dictating how many points the ADC will collect
        readout_duration = Tacq * 1E6
        N = int(readout_duration / self.dwell_time)
        if N % 2 == 1:
            N = N + 1
        readout_duration = self.dwell_time * N
        self.readout_duration = readout_duration
        
        # Ceil to map duration onto the scanner's 10us system clock raster
        self.plateau_time = self._rounduptime(readout_duration)

    def _gradientShape(self, t):
        """
        Calculates the instantaneous target k-space position at time parameter t.
        
        Uses the analytical spiral formula mapping a single spiral interleave.
        
        Parameters
        ----------
        t : float or np.ndarray
            Unitless timeframe scaling from 0 to 1 over the plateau duration.
            
        Returns
        -------
        kspace : np.ndarray
            Target spatial Cartesian coordinate at time t. 
        """
        time = t * self.plateau_time / 1E6

        # Support querying the analytical phase based on a singular timeframe constant
        if (isinstance(t, float) or isinstance(t, int)):
            if time < self.Ts:
                theta = self._theta1(t)
            else:
                theta = self._theta2(t)
        # Support querying piecewise temporal segments across an entire gradient evolution array
        else:
            idx_isSlewing = np.squeeze(np.argwhere(time < self.Ts))
            idx_isNotSlewing = np.squeeze(np.argwhere(time >= self.Ts))
            theta = np.zeros_like(time)
            theta[idx_isSlewing] = self._theta1(t[idx_isSlewing])
            theta[idx_isNotSlewing] = self._theta2(t[idx_isNotSlewing])

        k = self.lamda/10 * theta * self.matrix_size
        dirX = np.array([1,0,0])
        dirY = np.array([0,1,0])
        
        # Match output tensor dimensionality with array inputs if t was an ndarray
        if not (isinstance(t, float) or isinstance(t, int)):
            N = len(t)
            dirX = np.stack(N*[dirX], axis=1)
            dirY = np.stack(N*[dirY], axis=1)

        return dirX * k * np.cos(theta + self.shot_angle) + dirY * k * np.sin(theta + self.shot_angle)

    def _theta1(self, t):
        """
        Calculates the spiral phase angle theta for the slew-rate limited regime.
        
        Parameters
        ----------
        t : float or np.ndarray
            Parameterization timeframe (0 to 1).
            
        Returns
        -------
        theta : float or np.ndarray
            Accumulated spiral radial angle (rad) during initial acceleration.
        """
        time = t * self.plateau_time / 1E6
        output = (0.5 * self.beta * time**2) / (1 + self.beta/(2*self.a2) * time**(4.0/3))
        return output

    def _theta2(self, t):
        """
        Calculates the spiral phase angle theta for the amplitude limited regime.
        
        Only applicable in the outer region of k-space after the gradient reaches
        the scanner's maximum allowable amplitude magnitude.
        
        Parameters
        ----------
        t : float or np.ndarray
            Parameterization timeframe (0 to 1).
            
        Returns
        -------
        theta : float or np.ndarray
            Accumulated spiral radial angle (rad) during terminal velocity.
        """
        time = t * self.plateau_time / 1E6
        gamma = self.gamma*1E2
        maxamp = self.maxamp/10
        output = (self.theta_s**2 + 2*gamma/self.lamda*maxamp*(time - self.Ts))**(0.5)
        return output
