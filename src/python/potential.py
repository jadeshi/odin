
"""
potential.py

Experimental Data & Simulation Synthesis.

This file contains classes necessary for the integration of simulation and
experimental data. Specifically, it serves a Potential class used (by md.py
or mc.py) to run simulations. Further,

"""

import abc

import numpy as np

from mdtraj import Trajectory
from mdtraj import utils as mdutils

from odin import exptdata

import logging
logging.basicConfig()
logger = logging.getLogger(__name__)
#logger.setLevel('DEBUG')


class Potential(object):
    """
    Attributes for a kind of experimental potential. Any potential that
    inherets from this class can be sampled (integrated) by the samplers in
    odin/sample.py.
    """
    
    __metaclass__ = abc.ABCMeta
    
    @abc.abstractmethod    
    def __call__(self, trajectory):
        """
        Takes a set of xyz coordinates and evaluates the potential on that
        conformation.
        
        Parameters
        ----------
        trajectory : mdtraj.Trajectory
            A trajectory to evaluate the potential at
        """    
        return energy
    
    
    def _check_is_traj(self, trajectory):
        """
        ensure an `trajectory` object faithfully represents an atomic configuration
        """
        # typecheck
        if not type(trajectory) == Trajectory:
            raise TypeError('`trajectory` must be type mdtraj.Trajectory, got: %s' % type(trajectory))
        return
    
    
class FlatPotential(Potential):
    """
    This is a minimal implementation of a Potenial object, used mostly for testing.
    It can also be used to integrate a model in a prior potential only, without
    any experimental information.
    """
    
    def __call__(self, traj):
        """
        Takes a set of xyz coordinates and evaluates the potential on that
        conformation.
        
        Parameters
        ----------
        trajectory : mdtraj.Trajectory
            A trajectory to evaluate the potential at
        """
        self._check_is_traj(traj)
        return np.ones(traj.n_frames)
    
        
class WeightedExptPotential(Potential):
    """
    This class implements a potential of the form:
    
        V(x) = sum_i { lambda_i * f_i(x) }
        
    -- lambda_i is a scalar weight
    -- f_i is an experimental prediction for conformation 'x'
    """
    
    def __init__(self, *experiments):
        """
        Initialize WeightedExptPotential.
        
        Parameters
        ----------
        *args : odin.exptdata.ExptDataBase
            Pass any number of experimental data sets, all of which are combined
            into the weighted potenial.
        """
        
        self._experiments = []
        self._num_measurements = 0
        self._weights = np.array([])
        
        for expt in experiments:
            self.add_experiment(expt)
            
        return
    
        
    def __call__(self, trajectory):
        """
        Takes a set of xyz coordinates and evaluates the potential on that
        conformation.
        
        Parameters
        ----------
        trajectory : mdtraj.Trajectory
            A trajectory to evaluate the potential at
        """
        self._check_is_traj(trajectory)
        energy = np.sum( self.weights[None,:] * self.predictions(trajectory), axis=1 )
        return energy
    
        
    def add_experiment(self, expt):
        """
        Add an experiment to the potential object.
        
        Parameters
        ----------
        expt : odin.exptdata.ExptDataBase
            An experiment.
        """
        if not isinstance(expt, exptdata.ExptDataBase):
            raise TypeError('each of `experiments` must inheret from odin.exptdata.ExptDataBase')
        self._experiments.append(expt)
        self._num_measurements += expt.num_data
        self._weights = np.concatenate([ self._weights, np.ones(expt.num_data) ])
        assert len(self._weights) == self._num_measurements
        return
        
        
    @property
    def weights(self):
        return self._weights
    
        
    @property
    def num_measurements(self):
        return self._num_measurements
    
        
    @property
    def num_experiments(self):
        return len(self._experiments)
    
        
    def set_all_weights(self, weights):
        """
        Set the weights for all the experiments.
        
        Parameters
        ----------
        weights : np.ndarray
            An array of the weights to use. Must be self.num_measurements long.
        """
        if not type(weights) == np.ndarray:
            raise
        if not len(weights) == self.num_measurements:
            raise ValueError('`weights` must be len self.num_measurements. Got'
                             ' len: %d, require: %d' % (len(weights), 
                                                        self.num_measurements))
            
        self._weights = weights
        return
    
        
    def expt_weights(self, expt_index):
        """
        Get the weights corresponding to a single experiment.
        
        Parameters
        ----------
        expt_index : int
            The index corresponding to the experiment to get weights for.
        """
        
        start = 0
        for i,expt in enumerate(self._experiments):
            if i == expt_index:
                end = start + expt.num_data
                break
            else:
                start += expt.num_data
        
        logger.debug('start/end: %d/%d' % (start, end))
        
        return self._weights[start:end]
    
        
    def predictions(self, trajectory):
        """
        Method to predict the array `values` for each snapshot in `trajectory`.
        
        Parameters
        ----------
        trajectory : mdtraj.trajectory
            A trajectory to predict the experimental values for.
        
        Returns
        -------
        prediction : ndarray, 2-D
            The predicted values. Will be two dimensional, 
            len(trajectory) X len(values).
        """
        
        predictions = np.array([[]])
        for expt in self._experiments:
            predictions = np.concatenate([ predictions, expt.predict(trajectory) ], axis=1)
        
        assert predictions.shape[0] == trajectory.n_frames
        assert len(predictions.shape) == 2
        
        return predictions
    
        