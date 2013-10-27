
"""
tests for potential.py
"""

import mdtraj

import numpy as np
from numpy.testing import (assert_almost_equal, assert_array_almost_equal,
                           assert_allclose, assert_array_equal)
                           
from odin import potential
from odin import exptdata
from odin.testing import ref_file

def test_flat_potential():
    fp = potential.FlatPotential()
    ala2 = mdtraj.trajectory.load(ref_file('ala2.pdb'))
    assert_array_equal(fp(ala2), np.ones(1))

class TestWeightedExptPotential(object):
    
    def setup(self):
        
        # a dummy expts for testing
        restraint_array = np.zeros((2,4))
        restraint_array[0,:] = np.array([0, 5,   1.0, 1])
        restraint_array[1,:] = np.array([4, 10, 10.0, 0])
        self.expt1 = exptdata.DistanceRestraint(restraint_array)
        
        restraint_array = np.zeros((3,4))
        restraint_array[0,:] = np.array([0, 5,   1.0, 1])
        restraint_array[1,:] = np.array([4, 10, 10.0, 0])
        restraint_array[2,:] = np.array([4, 10, 10.0, 0])
        self.expt2 = exptdata.DistanceRestraint(restraint_array)
        
        self.num_meas = 5
        self.default_weights = np.ones(self.num_meas)
        self.wep = potential.WeightedExptPotential(self.expt1, self.expt2)
        
        return
        
    def test_call(self):
        ala2 = mdtraj.trajectory.load(ref_file('ala2.pdb'))
        energy = self.wep(ala2)
        assert energy[0] == 3.0
        assert energy.shape == (1,)
        
    def test_add_experiment(self):
        self.wep.add_experiment(self.expt1)
        assert self.wep.num_experiments == 3
        
    def test_weights(self):
        assert_array_almost_equal(self.wep.weights, self.default_weights)
        
    def test_num_experiments(self):
        assert self.wep.num_experiments == 2
        
    def test_set_all_weights(self):
        self.wep.set_all_weights(np.zeros(self.num_meas))
        assert_array_almost_equal(self.wep.weights, np.zeros(self.num_meas))
        
    def test_expt_weights(self):
        assert len(self.wep.expt_weights(0)) == 2
        assert len(self.wep.expt_weights(1)) == 3
        
    def test_predictions(self):
        ala2 = mdtraj.trajectory.load(ref_file('ala2.pdb'))
        p = self.wep.predictions(ala2)
        assert p.shape == (1,5)
        #print 'pred0', self.expt1.predict(ala2)
        #print 'prediction', p
        
        
        
        