
"""
models.py

Models odin can create. These manage the backend statistical integration (MCMC)
and can also generate requests for more simulation.
"""

import abc


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
        return predictions

    # -----------------
    # dealing with remote workers/sampling
    
    def connect_remote(self, remote):
        pass

    @property
    def num_remotes(self):
        pass
        
    def _sample(self, potential, n_steps):

        for i in range(self.num_remotes):
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
        return dictionary
        
    # -----------------
    # file handling
    
    @classmethod
    def load(cls, filename):
        pass
        
    
        
    


class SingleStructureModel(StructuralModel):
    
    def __init__():
        pass
    
        
class EnsembleModel(StructuralModel):
    
    def __init__():
        pass
        
    def converge_potential(self):

        potential = starting_potential

        while not potential.converged:
            while round_not_converged:
                self._sample_potential(potential, n_steps)
            
                # need to think about how to do the function below
                round_not_converged = evaluate_round_convergence()
            
            
            potential.optimize_lambdas()