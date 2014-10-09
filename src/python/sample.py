
"""
sample.py

Methods for sampling custom potentials.
"""

import sys
import numpy as np

import mdtraj
from mdtraj import reporters

from odin.potential import Potential

import logging
logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel('DEBUG')

try:
    from simtk.openmm import app
    import simtk.openmm as mm
    from simtk import unit
except ImportError as e:
    logger.critical(e)
    raise ImportError('You must have OpenMM installed to employ the sampler '
                      '(https://simtk.org/home/openmm).')


# ---------------------------------
# universal constants
k_boltz = 0.0083144621 # kJ / mol-K
# ---------------------------------

class MCReporter(reporters.HDF5Reporter):
    """
    A light wrapper around the HDF5 reporter that only reports if a Monte
    Carlo algorithm accepts a move.
    
    Does this by checking the MC state (accept vs. reject), and if the state
    is "reject", then it returns `None`s.
    """
    
    # overwrite
    def report(self, simulation, state, accept_move=True):
        """
        Generate a report.
        
        Parameters
        ----------
        simulation : simtk.openmm.app.Simulation
            The Simulation to generate a report for
            
        state : simtk.openmm.State
            The current state of the simulation
            
        accept_move : bool
            Whether or not the MC move is valid
        """
        if accept_move:
            super(MCReporter, self).report(simulation, state)
        else:
            pass # do nothing
        return
        
    # def describeNextReport(self, simulation):
    #     """
    #     Get information about the next report this object will generate.
    #     
    #     Parameters
    #     ----------
    #     simulation : simtk.openmm.app.Simulation
    #     The Simulation to generate a report for
    # 
    #     Returns
    #     -------
    #     report_description : tuple
    #     A five element tuple. The first element is the number of steps
    #     until the next report. The remaining elements specify whether
    #     that report will require positions, velocities, forces, and
    #     energies respectively.
    #     """
    #     steps = self._reportInterval - simulation.currentStep % self._reportInterval
    #     return (steps, self._coordinates, self._velocities, False, self._needEnergy)
    

class MDMC(object):
    """
    Performs hybrid MD/Monte-Carlo, where a short MD run is a moveset.
    """

    def __init__(self, potential, prior, topology, starting_positions,
                 scaling_speed=1.1, target_accept_percent=0.75,
                 steps_per_iter=1000, temperature=300.0,
                 openmm_platform='CUDA', 
                 platform_properties={'CudaPrecision': 'mixed'}):
        """
        Initialize an Openmm Monte Carlo sampler.
        
        Parameters
        ----------
        potential: odin.potential.Potential
            The potential to sample in. Should be a callable object that when
            passed an xyz conformation (array or mdtraj.trajectory) returns
            the energy for each snapshot.
            
        prior: str
            The prior potential to use. Must be a forcefield implemented in
            OpenMM.
        """
        
        # potential information
        if not isinstance(potential, Potential):
            raise TypeError('Argument `potential` must be of type '
                            'odin.potential.Potential')
        self.potential = potential
        self.set_prior(prior)
        
        # monte carlo options
        self.temperature = temperature
        if scaling_speed <= 1.0:
            raise ValueError('`scaling_speed` must be a float >= 1.0')
        self.scaling_speed = scaling_speed
        self.target_accept_percent = target_accept_percent
        self.steps_per_iter = steps_per_iter # number of MD steps per MC attempt
        
        self.total_moves_attempted = 0  # total number of MC accept/rejects
        self.accepted = 0               # accepted of total_moves_attempted
        
        self.mc_length_increment = 5000 # all MC runs move in increments of this
        
        # positions/topology of system
        if isinstance(topology, mdtraj.Topology):
            self.topology = topology
        elif isinstance(topology, app.topology.Topology):
            self.topology = mdtraj.Topology.from_openmm(topology)
        else:
            raise TypeError('`topology` must be type: mdtraj.Topology')

        self.starting_positions = np.array(starting_positions)
        self.positions = starting_positions
        
        # OpenMM Options
        self._platform = openmm_platform
        self._properties = platform_properties
        self._simulation_initialized = False
        
        # initialize the simulation
        self._initialize_simulation()
        
        return
    
    @property
    def _openmm_attrs(self):
        """
        Return a list of OpenMM private attributes the class contains.
        """
        return ['_integrator', '_simulation', '_system']

    
    def __getstate__(self):
        """
        Prepare object for pickling.
        """

        cpy = self.__dict__.copy()

        # version 1 -- use xml
        #cpy._integrator = mm.XmlSerializer.serialize(self._integrator)
        #cpy._simulation = mm.XmlSerializer.serialize(self._simulation)
        #cpy._system = mm.XmlSerializer.serializeSystem(self._system)

        # version 2 -- delete and re-initialize upon unpickle
        del cpy['_integrator']
        del cpy['_simulation']
        del cpy['_system']
        del cpy['_forcefield']

        return cpy

        
    def __setstate__(self, state):
        """
        Gets called upon unpickling. `state` is the cpy obj above
        """
        # version 1 above
        #state._integrator = mm.XmlSerializer.deserialize(state._integrator)
        #state._simulation = mm.XmlSerializer.deserialize(state._simulation)
        #state._system = mm.XmlSerializer.deserializeSystem(state._system)
        #self.__dict__ = state

        # version 2
        self.__dict__ = state
        self.set_prior(self._prior_xml)
        self._initialize_simulation()

        return
        
    
    def set_prior(self, prior):
        """
        Set the prior potential. If you don't want a prior, use 'IdealGas'.
        
        Options include, but are not limited to:
        
        -- IdealGas
        -- AmberMinimal
        -- Amber99sbildn
        
        Parameters
        ----------
        prior: str
            The prior potential to use. Must be a forcefield implemented in
            OpenMM.
        """
        
        try:
            self._prior_xml = prior
            self._forcefield = app.ForceField(self._prior_xml)
        except Exception as e:
            logger.critical(e)
            raise RuntimeError('Failed to initialized Forcefield. Most likely the'
                               ' prior potential you requested, %s, does not exist'
                               ' in the local OpenMM implementation.' % prior)
        return
        
        
    def _initialize_simulation(self):
        """
        """

        self._system = self._forcefield.createSystem(self.topology.to_openmm(),
                                               nonbondedMethod=app.CutoffNonPeriodic,
                                               nonbondedCutoff=1.0*unit.nanometers,
                                               constraints=app.HBonds,
                                               rigidWater=True, 
                                               ewaldErrorTolerance=0.0005)
            
        self._integrator = mm.LangevinIntegrator(300*unit.kelvin, 
                                                 1.0/unit.picoseconds, 
                                                 2.0*unit.femtoseconds)
        self._integrator.setConstraintTolerance(0.00001)
        
        mm.Platform.getPlatformByName(self._platform)
        self._simulation = app.Simulation(self.topology.to_openmm(),
                                          self._system,
                                          self._integrator) 
                                          #self._platform)
                                         # self._properties)
        self._simulation.context.setPositions(self.positions)

        self._simulation.minimizeEnergy()
        self._simulation.context.setVelocitiesToTemperature(300*unit.kelvin)
        
        return
    
        
    def sample(self, num_moves, output_target):
        """
        Run the sampler for `num_moves`.
        
        Parameteres
        -----------
        num_moves : int
            The number of moves to generate.
            
        output_target : file
            A file object that serialized results will be sent to. Could be
            a pipe or something similar.
            
        Returns
        -------
        sample : mdtraj.trajectory
            An `num_moves` length trajectory, sampled from the potential
            assoicated with the 
        """
        
        # need to figure out how we're going to deal with the traj data -- cant
        # keep it all in memory. Would be good to dump it into a single h5 db
        reporter = reporters.HDF5Reporter(output_target, 
                                          self.mc_length_increment, 
                                          coordinates=True, time=False, cell=False,
                                          potentialEnergy=False, kineticEnergy=False,
                                          temperature=False, velocities=False,
                                          atomSubset=None)
        self._simulation.reporters.append(reporter)

        # perform the actual monte carlo
        current_energy = self.potential(self.positions)
        
        for mi in range(num_moves):
            
            # step forward in time
            self.total_moves_attempted += 1
            self._simulation.step(self.steps_per_iter)

            # accept or reject according to Metropolis
            new_energy = self.potential( np.array(self._simulation.context.getState(getPositions=True).getPositions()) )
            
            # accept
            logger.debug('\tcurrent energy: %f' % current_energy)
            logger.debug('\tnew energy:     %f' % new_energy)
            if (new_energy < current_energy) or (np.random.rand() < np.exp( (current_energy - new_energy) / (k_boltz * self.temperature)) ):
                
                logger.debug('\t\tmove accepted')
                self.accepted += 1
                moves_done += 1
                self.positions = self._simulation.context.getState(getPositions=True).getPositions()
                
            # reject
            else:
                logger.debug('\t\tmove rejected')
                
            # perform adaptive update on the number of steps performed per attempt
            # to try and get to the `target_accept_percent` acceptance ratio
            self.accepted_ratio = float(self.accepted) / float(self.total_moves_attempted)
            if self.accepted_ratio > self.target_accept_percent:
                self.steps_per_iter /= self.scaling_speed
            else:
                self.steps_per_iter *= self.scaling_speed
            self.steps_per_iter = int(self.steps_per_iter)
                
            self.steps_per_iter -= self.steps_per_iter % self.mc_length_increment
            self.steps_per_iter = max(self.steps_per_iter, self.mc_length_increment)
            logger.debug('Current `steps_per_iter`: %d' % self.mc_length_increment) 
        
        # close repoter
        reporter.close()

        return
