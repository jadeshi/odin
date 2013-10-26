
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
           """
           xyz = np.array(xyz)
           if len(xyz.shape) == 3:
               pass
           elif len(xyz.shape) == 2:
               xyz = np.expand_dims(xyz, axis=0)
           else:
               raise TypeError('`xyz` must be a 2 or 3 dimensional array')
           return np.ones(xyz.shape[0])
    
        
class WeightedExptPotential(Potential):
    """
    
    """
    
    def __call__(self, xyz):
        return
        
        
    @property
    def weights(self):
        return self._weights
        
    def set_weights(self, weights):
        # checks
        self._weights = weights