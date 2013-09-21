
"""
Tests for /src/python/parse.py
"""

import os, sys
import warnings
from nose import SkipTest

import numpy as np
from numpy.testing import assert_allclose

from odin import xray
from odin.xray import parse
from odin.testing import skip, ref_file, gputest, expected_failure

from mdtraj import trajectory, io


class TestCBF(object):
    
    def setup(self):
        self.cbf = parse.CBF(ref_file('test_cbf.cbf'))
        
    def test_intensities_shape(self):
        s = self.cbf.intensities_shape
        assert s == (2527, 2463)
        
    def test_pixel_size(self):
        x = self.cbf.pixel_size
        assert x == (0.000172, 0.000172)
        
    def test_path_length(self):
        l = self.cbf.path_length
        assert l == 0.18801
        
    def test_wavelength(self):
        l = self.cbf.wavelength
        assert l == 0.7293
        
    def test_polarization(self):
        p = self.cbf.polarization
        assert p == 0.99
    
    def test_center(self):
        c = self.cbf.center
        ref = np.array([ 1264.63487097,  1231.26068894 ])
        #ref = np.array([ 1264.61906, 1231.24931508]) # this was using dermens algo
        assert_allclose(c, ref, rtol=1e-03)
    
    def test_corner(self):
        c = self.cbf.corner
        x = self.cbf.pixel_size
        y = self.cbf.center
        ref = (-x[1] * float(y[1]), -x[0] * float(y[0]))
        print ref, c
        assert ref == c
        

class TestEDF(object):
    
    def setup(self):
        self.edf = parse.EDF(ref_file('test_edf.edf'))

    def test_intensities_shape(self):
        s = self.edf.intensities_shape
        assert s == (619, 487)
        
    def test_numpix(self):
        print self.edf.num_pixels
        assert self.edf.num_pixels == 301453

    def test_pixel_size(self):
        x = self.edf.pixel_size
        assert x == (0.000172, 0.000172)
    
    def test_center(self):
        c = self.edf.center
        ref = np.array([ 308.33235619,  243.66551465 ])
        assert_allclose(c, ref, rtol=1e-03)
    
    def test_corner(self):
        c = self.edf.corner
        x = self.edf.pixel_size
        y = self.edf.center
        ref = (-x[1] * float(y[1]), -x[0] * float(y[0]))
        print ref, c
        assert ref == c
        
        
class TestTIFF(object):
    
    def setup(self):
        self.tiff = parse.TIFF(ref_file('test.tif'))

    def test_intensities_shape(self):
        s = self.tiff.intensities_shape
        assert s == (2048, 2048)
        
    def test_numpix(self):
        print self.tiff.num_pixels
        assert self.tiff.num_pixels == 2048*2048
    
    # current test data is just noise
    # so no def test_center(self):
        
        
class TestCXI(object):
    def setup(self):
        pass
