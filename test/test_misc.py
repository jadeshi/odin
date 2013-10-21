
"""
Test misc extension code.
"""

import numpy as np
from numpy.testing import *

from odin import misc_ext as misc
from odin import xray

class TestSolidAngle(object):
    
    def setup(self):
        self.d = xray.Detector.generic()
        self.i = np.ones(self.d.num_pixels)
        return
        
    def test_fast_sac(self):
        sa = misc.SolidAngle(self.d, use_fast_approximation=True)
        out = sa(self.i)
        
    def test_rigorous_sac(self):
        sa = misc.SolidAngle(self.d, use_fast_approximation=False)
        out = sa(self.i)
        
    def test_fast_vs_rigorous(self):
        saf = misc.SolidAngle(self.d, use_fast_approximation=True)
        sar = misc.SolidAngle(self.d, use_fast_approximation=False)
        fi = saf(self.i)
        ri = sar(self.i)
        percent_diff = float(np.sum( np.abs((fi / fi[0]) - (ri / ri[0])) < 0.2 )) / float(self.d.num_pixels)
        assert percent_diff < 0.02