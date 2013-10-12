
"""
potential.py

Experimental Data & Simulation Synthesis.

This file contains classes necessary for the integration of simulation and
experimental data. Specifically, it serves a Potential class used (by md.py
or mc.py) to run simulations. Further,

"""

import abc

from mdtraj import Trajectory
from mdtraj import utils as mdutils

class Potential(object):
    """
    Attributes for a kind of experimental potential. Any potential that
    inherets from this class can be sampled (integrated) by the samplers in
    odin/sample.py.
    """
    
    __metaclass__ = abc.ABCMeta
    
    @abc.abstractmethod    
    def __call__(self, xyz):
        """
        Takes a set of xyz coordinates and evaluates the potential on that
        conformation.
        """    
        return energy
    
        
class SingleParticlePotential(Potential):
    
    def __init__(self, expt_data, ):
        
        
        return
    
        
    def __call__(self, xyz):
        """
        Evaluate the potential of `xyz`.
        
        Parameters
        ----------
        xyz : mdtraj.Trajectory, np.ndarray
            Either an MD trajectory or 3-d array describing (snapshots, atoms, xyz)
            
        Returns
        -------
        potential : np.ndarray, float
            The potential for each frame of `xyz`
        """
        
        if isinstance(xyz, trajectory.Trajectory):
            coords = xyz.xyz
        elif:
            mdutils.ensure_type(xyz, np.float, 3, 'xyz', add_newaxis_on_deficient_ndim=True)
            coords = xyz
        else:
            raise TypeError('`xyz` must be type {mdtraj.Trajectory, np.ndarray}')
            
        preds = self._expt_data_collection.predict(xyz)
        potential = np.exp( 1/spreds

        
class EnsemblePotential(Potential):
    """
    A posterior potential that enforces a set of experimental constraints.
    """

        
    @property
    def lambdas_converged(self):
        return self._evaluate_convergence()
        
        
    def optimize_lambdas(algorithm='default'):
        """
        This is the only place where the experimentally measured data actually
        gets used.
        """
        pass
    

    def _hessian(self):
        
        pass
        
        
    def _evaluate_convergence(self):
        pass
        

    def fi(self, xyz):
        """
        Compute all the f_i values for a configuration `xyz`
        """
        pass