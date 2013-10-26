
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
        
        Parameters
        ----------
        xyz : np.ndarray
            Either a 2 or 3 dimensional array corresponding to a configuration 
            or set of configurations, respectively.
        """    
        return energy
    
    
    def _check_xyz(self, xyz):
        """
        ensure an `xyz` object faithfully represents an atomic configuration
        """
        
        # typecheck
        if type(xyz) == mdtraj.Trajectory:
            xyz = xyz.xyz
        elif type(xyz) != np.ndarray:
            try:
                xyz = np.array(xyz)
            except Exception as e:
                logger.critical(e)
                raise TypeError('`xyz` must be type {ndarray, mdtraj.Trajectory}'
                                ', got: %s' % str(type(xyz)))
               
        # dimension check
        if len(xyz.shape) == 3:
            pass
        elif len(xyz.shape) == 2:
            xyz = np.expand_dims(xyz, axis=0)
        else:
            raise TypeError('`xyz` must be a 2 or 3 dimensional array')
        
        return xyz
    
    
class Prior(Potential):
    """
    This is a minimal implementation of a Potenial object, used mostly for testing.
    It can also be used to integrate a model in a prior potential only, without
    any experimental information.
    """
    
    def __call__(self, xyz):
        """
        Takes a set of xyz coordinates and evaluates the potential on that
        conformation.
        
        Parameters
        ----------
        xyz : np.ndarray
            Either a 2 or 3 dimensional array corresponding to a configuration 
            or set of configurations, respectively.
        """
        xyz = self._check_xyz(xyz)
        return np.ones(xyz.shape[0])
    
        
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
        self.weights = np.array([])
        
        for expt in experiments:
            self.add_experiment(expt)
            
        return
    
        
    def __call__(self, xyz):
        """
        Takes a set of xyz coordinates and evaluates the potential on that
        conformation.
        
        Parameters
        ----------
        xyz : np.ndarray
            Either a 2 or 3 dimensional array corresponding to a configuration 
            or set of configurations, respectively.
        """
        xyz = self._check_xyz(xyz)
        energy = sum( self.weights * self.predictions(xyz) )
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
        for i,expt in self._experiments:
            if i == expt_index:
                end = start + expt.num_data
            else:
                start += expt.num_data
            
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
        
        predictions = np.array([])
        for expt in self._experiments:
            predictions = np.concatenate([ predictions, expt.predict(trajectory) ])
        
        assert len(predictions) == self._num_measurements
        
        return predictions
    
        