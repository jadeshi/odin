
"""
exptdata.py

Experimental Data
"""

import os
import abc
from glob import glob

import numpy as np
from mdtraj import io

import logging
logging.basicConfig()
logger = logging.getLogger(__name__)
 

class ExptDataBase(object):
    """
    Abstract base class for experimental data classes.
        
    All ExptData objects should have the following properties:
    -- values (the experimental values)
    -- errors (the STD error associated with values)
        
    Further, each ExptData inheretant should provide a self.predict(xyz) method,
    that outputs an array of len(values) that is the *prediction* of the array
    `values` given a molecular geometry `xyz`.
    """
    
    __metaclass__ = abc.ABCMeta
        
    # the following methods will be automatically inhereted, but are mutable
    # by children if need be
    
    @abc.abstractmethod
    def load(self, filename):
        """
        Load a file containing experimental data, and dump the relevant data,
        errors, and metadata into the object in a smart/comprehendable manner.
        """
        return
        
    @abc.abstractmethod
    def predict(self, trajectory):
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
        return prediction
    
        
    @abc.abstractmethod
    def log_likelihood(self, trajectory):
        """
        Compute the log_likelihood of each snapshot in `trajectory` given the
        specific model in question.
        
        Parameters
        ----------
        trajectory : mdtraj.Trajectory
            A trajectory of conformations to compute the log-likelihood of
            
        Returns
        -------
        log_likelihood : np.ndarray
            The log-likehood of each trajectory given the model & data
        """
        # NOTE: the implementation of this method will depend heavily on
        #       kind of experimental data used -- implementation will be in
        #       the next level of the class dependency tree
        return log_likelihood
    
        
    @abc.abstractmethod
    def prediction_log_likelihood(self, predictions):
        """
        Compute the log_likelihood of each snapshot in `trajectory` given the
        specific model in question.
        
        Parameters
        ----------
        predictions : np.ndarray
            The experimental predictions for each trajectory
            
        Returns
        -------
        log_likelihood : np.ndarray
            The log-likehood of each trajectory given the model & data
        """
        # NOTE: the implementation of this method will depend heavily on
        #       kind of experimental data used -- implementation will be in
        #       the next level of the class dependency tree
        return log_likelihood
        
    
class EnsembleExpt(ExptDataBase):
    """
    A container object for a set of experimental data performed on an ensemble
    of structures.
    """
        
    __metaclass__ = abc.ABCMeta
            
    @property
    def n_data(self):
        return self._n_data
        
    @property
    def values(self):
        """
        The measured values of each data point
        """
        return self._get_values()
        
    @property
    def errors(self):
        """
        The errors associated with each experiment.
        """
        return self._get_errors()
        
    def log_likelihood(self, trajectory):
        """
        Compute the log_likelihood of each snapshot in `trajectory` given the
        specific model in question.
        
        Parameters
        ----------
        trajectory : mdtraj.Trajectory
            A trajectory of conformations to compute the log-likelihood of
            
        Returns
        -------
        log_likelihood : np.ndarray
            The log-likehood of each trajectory given the model & data
        """
        predictions = self.predict(trajectory)
        return self.prediction_log_likelihood(predictions)

    # Classes that inherent from ExptData must implement all the methods below 
    # this is enforced by the abstract class module
        
    @abc.abstractmethod
    def _default_error(self):
        """
        Method to estimate the error of the experiment (conservatively) in the
        absence of explicit input.
        """
        return error_guess
        
    @abc.abstractmethod
    def _get_values(self):
        """
        Return an array `values`, in an order that ensures it will match up
        with the method self.predict()
        """
        return values

    @abc.abstractmethod
    def _get_errors(self):
        """
        Return an array `errors`, in an order that ensures it will match up
        with the method self.predict()
        """
        return errors
        
        
class SingleMolecExperiment(ExptDataBase):
    """
    A container object for a set of experimental data performed on a single
    molecule
    """
    
    __metaclass__ = abc.ABCMeta
    
    # work in progress
    

class DistanceRestraint(EnsembleExpt):
    """
    An experimental data class that can be used for NOEs, chemical x-linking,
    etc. Anything that relies on a distance restraint.
    
    This class also serves as a basic test of its parent classes.
    
    The input here is an N x 4 array, with each row specifying
    
    row     entry
    ---     -----
     1  --> atom index 1
     2  --> atom index 2
     3  --> distance between atoms 1 & 2, in Angstroms
     4  --> restraint is satisfied in experiment or not (0 or 1)
        
    The function then returns a "1" if the restraint is satisfied, a "0" 
    otherwise. Note that by setting an experimental restraint that is not
    satisfied, you are saying the distance must be *greater* than the indicated
    value.
    """
        
    def __init__(self, restraint_array, errors=None):
        """
        Instantiate a DistanceRestraint experimental data class.
        
        An experimental data class that can be used for NOEs, chemical x-linking,
        etc. Anything that relies on a distance restraint.

        Parameters
        ----------
        restraint_array : np.ndarray, float
            An array in the following format:
            
                row     entry
                ---     -----
                 1  --> atom index 1
                 2  --> atom index 2
                 3  --> distance between atoms 1 & 2, in Angstroms
                 4  --> restraint is satisfied in experiment or not (0 or 1)

            The function then returns a "1" if the restraint is satisfied, a "0" 
            otherwise. Note that by setting an experimental restraint that is 
            not satisfied, you are saying the distance must be *greater* than  
            the indicated value. 
            
            Actually, so long as the error parameter is not set to zero, these
            can take fractional values representing the confidence with which
            they can be assigned.
            
        Optional Parameters
        -------------------
        errors : np.ndarray, float
            An array of the standard deviations representing the confidence in
            each restraint.
        """
        
        if not type(restraint_array) == np.ndarray:
            raise TypeError('`restraint_array` must be type np.ndarray, got '
                            '%s' % type(restraint_array))
        
        if not ((restraint_array.shape[1] == 4) and (len(restraint_array.shape) == 2)):
            raise ValueError('`restraint_array` must have shape (n,4), got '
                             '%s' % str(restraint_array.shape))
        
        self.restraint_array = restraint_array
        self._n_data = restraint_array.shape[0]
        
        if errors == None:
            self._errors = self._default_error()
        else:
            self._errors = errors
        
        return
    
        
    @classmethod
    def load(cls, filename):
        """
        Load a file containing experimental data, and dump the relevant data,
        errors, and metadata into the object in a smart/comprehendable manner.
        """
                          
        if filename.endswith('dat'):
            restraint_array = np.loadtxt(filename)
        else:
            raise IOError('cannot understand ext of %s, must be one of: '
                          '%s' % (filename, self.acceptable_filetypes))
        
        return cls(restraint_array)
    
        
    def predict(self, trajectory):
        """
        Method to predict the array `values` for each snapshot in `trajectory`.
        In this case, `values` is an array of 0's (distance bigger than
        restraint) and 1's (distance smaller than restraint)
        
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
        
        prediction = np.zeros( (trajectory.n_frames, self._n_data),
                                dtype=self.restraint_array.dtype )
        
        for i in range(trajectory.n_frames):
            for j in range(self._n_data):
                
                ai = int(self.restraint_array[j,0]) # index 1
                aj = int(self.restraint_array[j,1]) # index 2
                
                # compute the atom-atom dist and compare to the input (dist in ang)
                d = np.sqrt( np.sum( np.power(trajectory.xyz[i,ai,:] - \
                                              trajectory.xyz[i,aj,:], 2) ) ) * 10.0
                
                logger.debug('distance from atom %d to %d: %f A' % (ai, aj, d))
                                              
                if d > self.restraint_array[j,2]:
                    prediction[i,j] = 0.0
                else:
                    prediction[i,j] = 1.0
        
        return prediction
        
        
    def prediction_log_likelihood(self, predictions):
        """
        Compute the log_likelihood of each snapshot in `trajectory` given the
        specific model in question.
        
        Parameters
        ----------
        predictions : np.ndarray
            The experimental predictions for each trajectory
            
        Returns
        -------
        log_likelihood : np.ndarray
            The log-likehood of each trajectory given the model & data
        """
        
        # normal errors w/o correlation
        log_likelihood = np.log( 1.0 / (self.errors * np.sqrt(2.0 * np.pi)) ) -\
                         np.power(predictions - self.values, 2) / (2.0 * self.errors)
        
        return log_likelihood
    
        
    def _default_error(self):
        """
        Method to estimate the error of the experiment (conservatively) in the
        absence of explicit input.
        
        The errors are standard deviations around the mean, and thus are an
        one-D ndarray of len(values).
        """
        # special case -- since this expt value is binary
        return np.ones(self._n_data) * 0.1 # todo think of something better
    
        
    def _get_values(self):
        """
        Return an array `values`, in an order that ensures it will match up
        with the method self.predict()
        """
        values = self.restraint_array[:,3] # binary values
        return values
    
        
    def _get_errors(self):
        """
        Return an array `errors`, in an order that ensures it will match up
        with the method self.predict()
        """
        # this method is dumb for this class, see if we need it for others
        return self._errors
