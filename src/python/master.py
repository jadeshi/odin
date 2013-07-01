"""
This is a code sketch for a master process that will farm work out to a set of
'remotes', each of which will run a simulation.

Notes
-----
-- Thinking for a first attempt, work synchronously. Async can happen later
-- Need to find out if OpenMM can return trajectory objects to memory
-- Thinking iPython can handle the mapping to remotes

"""

class Master(object):
    
    def __init__(self):
        pass
        
        
    def run(self):
        
        self.converge_potential()
        self.sample()
        
        return
        
    
    @property
    def num_remotes(self):
        pass
    
        
    def register_remotes(self):
        pass
    
        
    def _sample_potential(self, potential, n_steps):
        
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
        
        
    def converge_potential(self):

        potential = starting_potential

        while not potential.converged:
            while round_not_converged:
                self._sample_potential(potential, n_steps)
            
                # need to think about how to do the function below
                round_not_converged = evaluate_round_convergence()
            
            
            potential.optimize_lambdas()