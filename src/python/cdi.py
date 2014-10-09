
"""
Test code for phasing a single view
"""

from odin import Potential
from thor import scatter


class CdiPotential(Potential):
    """
    Simple gaussian-error evaluator for CDI image
    """
    
    def __init__(intensities, qxyz, sigma, force_no_gpu=False):
        """
        Simple gaussian-error evaluator for CDI image
        
        Parameters
        ----------
        intensities : np.ndarray, float
            The observed intenisties, in a 1D array
            
        qxyz : np.ndarray, float
            The reciprocal space coordinates of the intensities, an N x 3 array
            where N is lenght of `intensities`.
            
        sigma : float
            The error on the experimental values (same for all pixels for now)
        """
        
        self.intensities = intensities
        self.qxyz = qxyz
        self.sigma = sigma
        
        self._no_gpu = force_no_gpu
        self._device_id = 0
        
        if not self.qxyz.shape == (len(intensities, 3)):
            raise ValueError('qxyz wrong shape')
            
        return
        
        
    def __call__(trajectory):
        """
        Return the system 'energy'.
        
        Parameters
        ----------
        """
        
        energies = np.zeros(trajectory.n_frames)
        
        for i in range(trajectory.n_frames)
            prediction = scatter.simulate_shot(trajectory[i], 1, self.qxyz,
                                               force_no_gpu=self._no_gpu, 
                                               device_id=self._device_id)
        
            energies[i] = np.sum( np.square(prediction - self.intensities) / 
                                  (2.0 * self.sigma) )
        
        return energies
    
    
    
            
        