
"""
models.py

Models odin can create. These manage the backend statistical integration (MCMC)
and can also generate requests for more simulation.
"""

import abc

import pymc
import numpy as np
import IPython as ip

from odin import potential


class StructuralModel(object):
    """
    Base class for structural models. This class contains abstract methods
    that should be implemented by specific kinds of structural models, but
    also has the implementation for how to work with remote workers. It also
    contains implementations for interacting with an HDF5 file that contains
    all the relevant information ODIN employs to construct a strutural model.

    Notes
    -----
    -- Thinking for a first attempt, work synchronously. Async can happen later
    -- Need to find out if OpenMM can return trajectory objects to memory
    -- Thinking iPython can handle the mapping to remotes
    """
    
    __metaclass__ = abc.ABCMeta
    
    
    def __call__(self, trajectory):
        return log_likelihood

    # -----------------
    # dealing with remote workers/sampling
    
    def connect_remote(self, remote):
        pass

    @property
    def num_remotes(self):
        pass
        
    def sample(self, client, num_structures):

        if not isinstance(client, IPython.parallel.Client)

        for i in range(len(client)):
            s = Simulation( Potential )
            s.set_positions( choose_random_structure() )
            sims.append(s)

        results = map_async(sims) # obviously this syntax isnt right

        # this needs to be smart
        # right now I'm thinking we collect the results back into local
        # memory and write them to disk. May be better to have the remotes
        # deal with that, though
        write_results(results)
        
    # -----------------
    # conformation loading, prediction
        
    def load_conformations(self, conformations):
        pass
    
    @property
    def conformations(self):
        return trajectories
        
    @property
    def predictions(self):
        return
        
    # -----------------
    # file handling
    
    @classmethod
    def load(cls, filename):
        pass
        
    
        
    


class SingleStructureModel(StructuralModel):
    
    def __init__():
        pass
    
        
class MaxEntEnsembleModel(StructuralModel, potential.WeightedExptPotential):
    """
    
    NOTES:
    
    This should compute (once) and store the expt predictions
    """
    
    def __init__(self, experiments, lambda_prior, nuts_options={}):
        """
        """
        
        
        # initialize empty containers
        self._conformations = None
        self._predictions = None
        
        self._experiments = []
        self._num_measurements = 0
        self._weights = np.array([])
        
        # deal w arguments of __init__
        # experiments
        for expt in experiments:
            self.add_experiment(expt)
            
        # lambda_prior
        if not isinstance(lambda_prior, pymc.distributions.distribution.Distribution):
            raise TypeError('self._prior_dist must be an instance of '
                            'pymc.distributions.distribution.Distribution, got:'
                            ' %s' % type(self._lambda_prior_dist))
        else:
            self._lambda_prior_dist = lambda_prior
            
        # nuts_options
        self.nuts_options = nuts_options
        
        return
    
        
    @property
    def lambda_prior_dist(self):
        if not isinstance(self._prior_dist, pymc.distributions.distribution.Distribution):
            raise TypeError('self._prior_dist must be an instance of '
                            'pymc.distributions.distribution.Distribution, got:'
                            ' %s' % type(self._lambda_prior_dist))
        return self._lambda_prior_dist
        
    # -------------------
    # methods from potential.WeightedExptPotential we may not want...
    #
    
    def __call__(self):
        return
    
    @property
    def weights(self):
        return self._weights
    
        
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
        
    # -------------------
        
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
        
        # This method overwrites the one inhereted from WeightedExptPotential
        # the only addition is that it implements a caching mechanism so that
        # we don't recompute the predictions if `trajectory` is unchanged
        
        if trajectory == self._conformations:
            predictions = self._predictions
        else:
            predictions = super(EnsembleModel, self).predictions(trajectory)
            self._predictions = predictions
        
        assert predictions.shape == (trajectory.n_frames, self.num_measurements)
        
        return
    
        
    def _probx_lambda(self, trajectory, lambdas):
        """
        Compute the Boltzmann weight of each snapshot in `trajectory` given
        the experimental weighting factors `lambdas`, usually written
        
            p_lambda(x) = 1/Z(lambda) * exp{ sum[ lambda_i * f_i(x) ] }
        
        Parameters
        ----------
        trajectory : mdtraj.Trajectory
            A trajectory of conformations to compute the probability of.
        
        lambdas : np.ndarray, float
            The Lagrange multipliers/exponential weights on the experimental
            predictions.
        
        Returns
        -------
        probs : np.ndarray, float
            The probability of each snapshot in `trajectory` given lambdas.
        """
        
        self._check_is_traj(trajectory)
        
        energy = np.sum( lambdas[None,:] * self.predictions(trajectory), axis=1 )
        assert energy.shape == (trajectory.n_frames,)
        
        probs = np.exp(-1.0 * energy)
        probs /= probs.sum()
        assert probs.shape == (trajectory.n_frames,)
        
        return probs

        
    def observable_predictions(self, lambdas, expt_index=None):
        """
        Computes and returns
        
                                   n
            <f_k(x)>_lambda = 1/n SUM{ f_k(x) * exp( SUM{ lambda_i * f_i(x_j) } ) }
                                  j=1                 i
                                  
        Parameters
        ----------
        lambdas : np.ndarray
            An array of the lagrange multipliers, lambda_i
            
        expt_index : int
            An index of the experiment to compute the predictions for. If `None`
            compute observable precitions for all experiments
            
        Returns
        -------
        avg_fi : np.ndarray
        """
        px = self._probx_lambda(self.conformations, lambdas)
        
        if expt_index == None:
            avg_fi = sum(px * self.predictions(self.conformations)[None,:], axis=1)
        elif expt_index > self.num_experiments-1:
            raise RuntimeError('Asked for the observable predictions from '
                               'experiment %d (zero-indexed), only %d '
                               'experiements in class.' % (expt_index > self.num_experiments))
        else:
            avg_fi = sum(px * self._experiments[expt_index].predictions(self.conformations)[None,:], axis=1)
            
        assert avg_fi.shape == (self.num_measurements,)
        return avg_fi
    
        
    @property
    def _lambda_posterior_model(self):
        """
        Return a pymc model for the posterior over lambda.
        """
        
        # warning: this block of code involves some trickery!
        
        # construct a model and add the prior over lambdas
        model = pymc.Model()
        lambda_prior = model.Var('prior', self.lambda_prior_dist)

        # we add each experiment as a "Data" type to the pymc model
        # here the "observables" (data values) are what we observed in the
        # experiment, and the parameters are a function of the current
        # MCMC sample of lambdas.        
        for expt in self._experiments:
                        
            dist = expt._error_model( self.observable_predictions(lambda_prior) )
            if not isinstance(pymc.distributions.distribution.Distribution, dist):
                raise TypeError('Incorrect return from %s._error_model: the'
                                ' experiments error model function does not'
                                ' work correctly. The implementation should'
                                ' return a tuple (dist, dist_params), see'
                                ' the documentation for more detail.' % str(expt))
            
            model.Data(expt.values, dist)
            
        return model
    
        
    def sample_lambda_posterior(self, num_steps):
        """
        """
        # right now regenerating the model object each time we want to sample
        # this will cost a little, but ensure we have the latest 
        # -- so could be sped up if we check to see what expts are in a chached
        #    model
        m = self._lambda_posterior_model()
        trace = m.sample(num_steps, pymc.NUTS(**self.nuts_params))
        return trace
    
        
    def _converge_lambdas(self):

        raise NotImplementedError()

        potential = starting_potential

        while not potential.converged:
            while round_not_converged:
                self._sample_potential(potential, n_steps)
            
                # need to think about how to do the function below
                round_not_converged = evaluate_round_convergence()
            
            
            potential.optimize_lambdas()
        return
        