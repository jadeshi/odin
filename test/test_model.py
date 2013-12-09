
"""
Tests: src/python/exptdata.py
"""

from odin import models
from odin import exptdata
from odin.testing import skip, ref_file, expected_failure
import mdtraj

import numpy as np
from numpy.testing import (assert_almost_equal, assert_array_almost_equal, 
                           assert_allclose, assert_array_equal)

# class TestStructuralModel(object):
#     # coming soon!
#     pass
    
class TestMaxEntEnsembleModel(object):
    
    def setup(self):
        
        restraint_array = np.zeros((2,4))
        restraint_array[0,:] = np.array([0, 5,   1.0, 1])
        restraint_array[1,:] = np.array([4, 10, 10.0, 0])
        dr = exptdata.DistanceRestraint(restraint_array)
        
        self.expts = [dr]
        self.confs = mdtraj.load(ref_file('ala3_3frames.h5'))
        self.meem = models.MaxEntEnsembleModel('IdealGas.xml', *self.expts)
        self.meem._conformations = self.confs # NEED TO REPLACE
                                               
        self.lambdas = np.ones(self.meem.num_measurements)
        
        return
                   
    # first pass : all smoke tests
    # need to make them real tests...
    @skip
    def test_snapshot_predictions(self):
        p = self.meem.snapshot_predictions(self.confs)
        print p
        
    @skip
    def test_probx_lambda(self):
        probs = self.meem._probx_lambda(self.confs, self.lambdas)
        print probs
        
    @skip
    def test_observable_predictions(self):
        ops = self.meem.observable_predictions(self.lambdas)
        print ops
        
    @skip
    def test_observable_predictions_single_expt(self):
        ops = self.meem.observable_predictions(self.lambdas, expt_index=0)
        print ops
    
    @skip
    def test_observable_prediction_covariance(self):
        cov = self.meem.observable_prediction_covariance(self.lambdas)
        print cov
        
    @skip
    def test_observable_prediction_covariance_single_expt(self):
        cov = self.meem.observable_prediction_covariance(self.lambdas, expt_index=0)
        print cov
        
    @skip
    def test_lambda_posterior_model(self):
        m = self.meem._lambda_posterior_model()
        print m.vars
        
    @skip
    def test_sample_lambda_posterior(self):
        l_samples = self.meem.sample_lambda_posterior(100)
        print l_samples
        
        
        
        
        