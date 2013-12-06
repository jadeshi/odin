
"""
models.py

Models odin can create. These manage the backend statistical integration (MCMC)
and can also generate requests for more simulation.
"""

import logging
logging.basicConfig()
logger = logging.getLogger(__name__)
#logger.setLevel('DEBUG')

import abc

import numpy as np
import IPython as ip

from odin import potential
from odin import utils

try:
    import pymc
except ImportError as e:
    logger.debug(e)
    PYMC_INSTALLED = False

try:
    from simtk.openmm import app
    import simtk.openmm as mm
    from simtk import unit
    OPENMM_INSTALLED = True
except ImportError as e:
    logger.debug(e)
    OPENMM_INSTALLED = False


class StructuralModel(object):
    """
    Base class for structural models. This class contains abstract methods
    that should be implemented by specific kinds of structural models, but
    also has the implementation for how to work with remote workers. It also
    contains implementations for interacting with an HDF5 file that contains
    all the relevant information ODIN employs to construct a strutural model.
    """
    
    __metaclass__ = abc.ABCMeta
    
    @abc.abstractmethod
    def __call__(self, trajectory):
        return log_likelihood

    # -----------------
    # dealing with remote workers/sampling
    
    def connect_remote(self, remote):
        raise NotImplementedError()

    @property
    def num_remotes(self):
        raise NotImplementedError()
        
    def sample(self, client, num_structures):
        
        raise NotImplementedError()

        if not isinstance(client, IPython.parallel.Client):
            pass

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
        raise NotImplementedError()
    
    @abc.abstractproperty
    def conformations(self):
        return trajectories
        
    @abc.abstractproperty
    def num_measurements(self):
        return num_measurements
        
    @abc.abstractproperty
    def snapshot_predictions(self):
        return
        
    @abc.abstractproperty
    def observable_predictions(self):
        return
        
    # -----------------
    # file handling
    @abc.abstractmethod
    # @classmethod
    def load(cls, filename):
        pass
    
    
    
class MaxEntEnsembleModel(StructuralModel, potential.ExptPotential):
    """
    Generate an ensemble model that gives the Boltzmann weights over a set of
    conformations that guarentees the ensemble (a) reproduces a set of 
    experimental observations (including error over measurements) while (b)
    choosing between degenerate options of (a) by minimizing the relative
    entropy with respect to some prior.
    
    Citations
    ---------
    ..[1] Kyle
    ..[2] TJ
    """
    
    def __init__(self, structural_prior, *experiments):
        """
        Initialize an instance of MaxEntEnsembleModel.
        
        Parameters
        ----------
        structural_prior : str OR np.ndarray
            Either this is a str defining a forcefield to employ as a prior
            (e.g. 'IdealGas.xml') or, if `initial_conformations` is supplied,
            this can be a 1D numpy array of the same length of the trajectory
            supplied there defining the prior population weights for each
            conformation.
        
        experiments : odin.exptdata.ExptDataBase
            Many experiments to include in the model.
            
        intial_conformations : mdtraj.Trajectory
            A set of conformations for the model to weight. Each will be
            assigned a Boltzmann probability when the model is trained.
            
        nuts_options : dict
            Kwargs for pymc's no-U turn sampler (NUTS) algorithm.
        """
        
        
        # initialize empty containers
        self._conformations = None
        self._predictions = None
        
        self._experiments = []
        self._num_measurements = 0
        self._weights = np.array([])
        
        # --- deal w arguments of __init__
        # structural_prior
        if type(structural_prior) == str:
            try:
                self._prior_xml = structural_prior
                self._forcefield = app.ForceField(self._prior_xml)
            except Exception as e:
                logger.debug(e)
                self._OPENMM = False
                logger.warning('Failed to initialized Forcefield. Most likely the'
                                ' prior potential you requested, %s, does not exist'
                                ' in the local OpenMM implementation, or you do not have OpenMM installed. Continuing, but you will need to supply pre-compp.' % structural_prior)
            else:
                self._OPENMM = OPENMM_INSTALLED
            
        elif type(structural_prior) == np.ndarray:
            if initial_conformations == None:
                raise Exception() 
            self._prior_xml  = None
            self._forcefield = None
            
        else:
            raise TypeError('`structural_prior` must be type {str, np.ndarray},'
                            ' got: %s' % type(structural_prior))
            
        # nuts_options
        self.nuts_options = {} # nuts_options
        
        # experiments
        for expt in experiments:
            self.add_experiment(expt) # from potential.ExptPotential
        
        return
    
        
    def __call__(self):
        """
        """
        raise NotImplementedError()
        return
    
    
    @property
    def structural_prior(self):
        raise NotImplementedError()
        return
        
        
    @property
    def conformations(self):
        return self._conformations
        
        
    @property
    def num_measurements(self):
        return self._num_measurements
    
        
    #@utils.memorize
    def snapshot_predictions(self, trajectory):
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
        
        # This method overwrites the one inhereted from ExptPotential
        # the only addition is that it implements a caching mechanism so that
        # we don't recompute the predictions if `trajectory` is unchanged
        
        if (trajectory == self._conformations) and (self._predictions != None):
            predictions = self._predictions
        else:
            predictions = super(MaxEntEnsembleModel, self).predictions(trajectory)
            self._predictions = predictions
        
        assert predictions.shape == (trajectory.n_frames, self.num_measurements)
        
        return predictions
    
        
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
        
        energy = np.sum( lambdas[None,:] * self.snapshot_predictions(trajectory), axis=1 )
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
            An array of the lagrange multipliers, lambda_i.
            
        expt_index : int
            An index of the experiment to compute the predictions for. If `None`
            compute observable precitions for all experiments.
            
        Returns
        -------
        avg_fi : np.ndarray
            A shape (num_measurements,) array of ensemble averaged predictions.
        """
        px = self._probx_lambda(self.conformations, lambdas)
        
        if expt_index == None:
            avg_fi = sum(px * self.snapshot_predictions(self.conformations)[None,:], axis=1)
        elif expt_index > self.num_experiments-1:
            raise RuntimeError('Asked for the observable predictions from '
                               'experiment %d (zero-indexed), only %d '
                               'experiements in class.' % (expt_index > self.num_experiments))
        else:
            avg_fi = sum(px * self._experiments[expt_index].predictions(self.conformations)[None,:], axis=1)
            
        assert avg_fi.shape == (self.num_measurements,)
        return avg_fi
    
    
    def observable_prediction_covariance(self, lambdas, expt_index=None):
        """
        Computes and returns the covariance matrix
        
            Sigma_ij = <f_i(x) f_j(x)>_lambda - <f_i(x)>_lambda * <f_j(x)>_lambda 
                                  
        Parameters
        ----------
        lambdas : np.ndarray
            An array of the lagrange multipliers, lambda_i.
            
        expt_index : int
            An index of the experiment to compute the predictions for. If `None`
            compute observable precitions for all experiments.
            
        Returns
        -------
        sigma : np.ndarray
            A shape (num_measurements,num_measurements) array of the prediction
            covariance matrix.
        """
        fi = self.snapshot_predictions()
        sigma = np.cov(fi)
        assert sigma.shape == (self.num_measurements, self.num_measurements)
        return sigma
        
        
    @property
    def _lambda_posterior_model(self):
        """
        Return a pymc model for the posterior over lambda.
        
        Returns
        -------
        model : pymc.model.Model
            A model object describing the Bayesian model for the posterior
            over lambdas.
        """
        
        # warning: this block of code involves some trickery!
        
        # construct a model
        model = pymc.Model()
        
        # we add each experiment as a "Data" type to the pymc model
        # here the "observables" (data values) are what we observed in the
        # experiment
               
        for i,expt in enumerate(self._experiments):
                        
            expt_pymc_model = expt._pymc_error_model() # is a dict
            
            if not 'likelihood' in expt_pymc_model.keys():
                raise RuntimeError('Could not find `likelihood` entry in the '
                                   'pymc model of experiment %d (%s). Please '
                                   'check the implementation of the error '
                                   'models of all experiments included in the'
                                   ' model.' % (i, str(expt)))
            
            # add all variables from the error model to the pyMC model
            for varname in expt_pymc_model:
                
                if not isinstance(pymc.distributions.distribution.Distribution,\
                                  expt_pymc_model[varname]):
                    raise TypeError('Incorrect return from %s._error_model: the'
                                    ' experiments error model function does not'
                                    ' work correctly. The implementation should'
                                    ' return a tuple (dist, dist_params), see'
                                    ' the documentation for more detail.' % str(expt))
                                    
                model.Var(varname + '-e' + str(i), expt_pymc_model[varname])
            
            # add the observed data
            model.Data(expt.values, dist)
            
            # generate a distribution "transform" that takes the likelihood in
            # observable-space and transforms it to lambda-space
            
            # NOTE: the 'fwd' function below is *not* correct, but it only is 
            # used for mean/mode calculations of the dist in pymc -- so long
            # as we avoid those, we should be OK
            fwd = lambda x : x # this is the wrong (but OK) one
            bkd = lambda x : self.observable_predictions(x, expt_index=i)
            J   = lambda x : np.abs(np.linalg.det(self.observable_prediction_covariance(x, expt_index=i)))
            TF = pymc.distributions.transforms.transform('posterior', fwd, bkd, J)
            
            # transform the error model from observable space to lambda space
            # and add it to the pymc model
            lambda_likehihood = TF.apply(dist)
            model.Var('lambda-likelihood'+ '-e' + str(i), lambda_likehihood)
            
        return model
    
        
    def sample_lambda_posterior(self, num_steps):
        """
        Sample values of lambda from the model posterior using the NUTS
        Hamiltonian MC algorithm. Performed with automatic tuning courtesy of
        pymc.
        
        Parameters
        ----------
        num_steps : int
            The number of MCMC steps to take.
            
        Returns
        -------
        trace : np.ndarray
            A shape-(num_steps, num_measurements) array with sequential samples
            of lambda corresponding to each experimental measurement.
        """
        
        # right now regenerating the model object each time we want to sample
        # this will cost a little, but ensure we have the latest 
        # -- so could be sped up if we check to see what expts are in a chached
        #    model
        
        m = self._lambda_posterior_model()
        trace = m.sample(num_steps, pymc.NUTS(**self.nuts_params))
        
        # stack the sampled lambdas for each experiment
        lambda_trace = np.concatenate( [ trace.samples[('lambda-likelihood'+ '-e' + str(i))].vals[0] for i in range(self.num_experiments) ], axis=1 )
        lambda_trace = np.squeeze(lambda_trace)
        
        assert lambda_trace.shape == (num_steps, self.num_measurements)
        
        return trace
    
        
    # def _converge_lambdas(self):
    # 
    #     raise NotImplementedError()
    # 
    #     potential = starting_potential
    # 
    #     while not potential.converged:
    #         while round_not_converged:
    #             self._sample_potential(potential, n_steps)
    #         
    #             # need to think about how to do the function below
    #             round_not_converged = evaluate_round_convergence()
    #         
    #         
    #         potential.optimize_lambdas()
    #     return
    
    
    @classmethod
    def load(self, filename):
        raise NotImplementedError()
        return
        