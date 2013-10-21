
"""
tests for odin/python/sample.py
"""

import os
import shutil
import tempfile

import numpy as np
from numpy.testing.decorators import skipif

import mdtraj
from mdtraj import HDF5TrajectoryFile
from mdtraj.testing import eq
from mdtraj.reporters import hdf5reporter
from mdtraj.reporters import HDF5Reporter

try:
    from simtk.unit import *
    from simtk.openmm import *
    from simtk.openmm.app import *
    HAVE_OPENMM = True
except ImportError:
    HAVE_OPENMM = False
        
from odin import sample
from odin import potential
from odin.testing import ref_file


testdir = tempfile.mkdtemp()
def teardown_module(module):
    """
    remove the temporary directory created by tests in this file
    this gets automatically called by nose
    """
    shutil.rmtree(testdir)
    

@skipif(not HAVE_OPENMM, 'No OpenMM')
def test_reporter():
    # stolen/modified from MDTraj/tests/test_reporter.py ... thanks rmcgibbo
    
    tempdir = os.path.join(testdir, 'test_reporter')
    os.makedirs(tempdir)

    pdb = PDBFile( ref_file('ala2.pdb') )
    forcefield = ForceField('amber99sbildn.xml', 'amber99_obc.xml')
    system = forcefield.createSystem(pdb.topology, 
                                    nonbondedMethod=CutoffNonPeriodic,
                                    nonbondedCutoff=1.0*nanometers,
                                    constraints=HBonds,
                                    rigidWater=True)
    integrator = LangevinIntegrator(300*kelvin, 1.0/picoseconds, 2.0*femtoseconds)
    integrator.setConstraintTolerance(0.00001)

    platform = Platform.getPlatformByName('Reference')
    simulation = Simulation(pdb.topology, system, integrator, platform)
    simulation.context.setPositions(pdb.positions)

    simulation.context.setVelocitiesToTemperature(300*kelvin)

    reffile  = os.path.join(tempdir, 'traj.h5')
    testfile = os.path.join(tempdir, 'traj-test.h5')

    ref_reporter = HDF5Reporter(reffile, 2, coordinates=True, time=True,
        cell=True, potentialEnergy=True, kineticEnergy=True, temperature=True,
        velocities=True)
    test_reporter = sample.MCReporter(testfile, 2, coordinates=True, time=True,
            cell=True, potentialEnergy=True, kineticEnergy=True, temperature=True,
            velocities=True)

    simulation.reporters.append(ref_reporter)
    simulation.reporters.append(test_reporter)
    simulation.step(100)

    ref_reporter.close()
    test_reporter.close()

    with HDF5TrajectoryFile(testfile) as f:
        got = f.read()
        yield lambda: eq(got.temperature.shape, (50,))
        yield lambda: eq(got.potentialEnergy.shape, (50,))
        yield lambda: eq(got.kineticEnergy.shape, (50,))
        yield lambda: eq(got.coordinates.shape, (50, 22, 3))
        yield lambda: eq(got.velocities.shape, (50, 22, 3))
        yield lambda: eq(got.time, 0.002*2*(1+np.arange(50)))
        yield lambda: f.topology == mdtraj.load(ref_file('ala2.pdb')).top

    ref_traj = mdtraj.load(reffile)
    test_traj = mdtraj.load(testfile)
    
    yield lambda: eq(ref_traj.xyz, test_traj.xyz)
    yield lambda: eq(ref_traj.unitcell_vectors, test_traj.unitcell_vectors)
    yield lambda: eq(ref_traj.time, test_traj.time)

    yield lambda: eq(ref_traj.xyz, test_traj.xyz)
    yield lambda: eq(ref_traj.unitcell_vectors, test_traj.unitcell_vectors)
    
        
class TestMDMC(object):
    
    def setup(self):
        self.potential = potential.Prior()
        self.prior = 'amber99sbildn.xml'
        self.pdb = mdtraj.load(ref_file('ala2.pdb'))
        self.topology = self.pdb.top
        self.starting_positions = self.pdb.xyz[0]
        self.mdmc = sample.MDMC(self.potential, self.prior, 
                                self.topology, self.starting_positions,
                                openmm_platform='reference', platform_properties={})
        
    @skipif(not HAVE_OPENMM, 'No OpenMM')
    def test_set_prior(self):
        # test all the priors included in the Odin repo
        prior_xmls = glob('priors/*.xml')
        for xml in prior_xmls:
            self.mdmc.set_prior(xml)
    
    @skipif(not HAVE_OPENMM, 'No OpenMM')
    def test_sample(self):
        if os.path.exists('test-traj.h5'): os.remove('test-traj.h5')
        self.mdmc.sample(100, 'test-traj.h5')
        t = mdtraj.trajectory.load('test-traj.h5')
        assert t.xyz[0] == 100
        os.remove('test-traj.h5')
        
        
        
        