import itertools
import os

import numpy as np
from scipy.stats import chi2 as chi2distribution
try:
    import matplotlib.pyplot as pp
    HAVE_PYPLOT = True
except ImportError:
    HAVE_PYPLOT = False

from odin.math2 import RandRot


        
def test_RandRot():
    "Test the RandRot() function"
    for t in _test_random_rotation(RandRot):
        yield t


def _test_random_rotation(rotation_matrix_factory, n_tests=200, n_bins=20):
    """Main test driver. Takes as input a function that generates random
    rotation matricies and tries rotating a bunch of vectors. In polar coordinates,
    we check via a chi2 test whether the polar angles and azimuthal angles
    appear to be distributed as they should.
    """

    v = np.array([1, 0, 0])
    rotated = np.empty((n_tests, 3))

    for i in range(n_tests):
        R = rotation_matrix_factory()
        rotated[i] = np.dot(v, R)
    r = np.sqrt(np.sum(np.square(rotated), axis=1))
    
    # polar angle theta (0, pi)
    theta = np.arccos(rotated[:, 2] / r)

    # azimuthal angle phi [0, 2pi)
    phi = np.arctan2(rotated[:,1], rotated[:,0])
    phi[phi < 0.0] += 2 * np.pi

    np.testing.assert_array_almost_equal(r, np.ones_like(r),
        err_msg='the length of each of the vectors should still be one')
    
    def assert_phi_chi2():
        # do a chi-squared test that the phi angles are uniform from [0, 2*pi)
        observed_counts, bin_edges = np.histogram(phi, bins=n_bins, range=(0, 2*np.pi))
        expected_counts = n_tests * np.diff(bin_edges) / (2*np.pi)
        assert_chisquared(observed_counts, expected_counts, bin_edges, title='phi (azimuthal angle)')

    def assert_theta_chi2():
        # marginalized over phi, the number of points at each polar angle theta
        # should follow a sin curve. The differential surface area element
        # is r^2 \sin \theta d\theta d\phi, so integrating over \phi gives us
        # dA(theta) = 2*pi * r**2 * sin(theta) * dtheta    
        # after binning, then we should have -2*pi*cos(b)
        observed_counts, bin_edges = np.histogram(theta, bins=n_bins, range=(0, np.pi))
        # the number of expected counts in each bin comes from integrating dA(theta)
        # over the theta bin, and then properly normalizing
        bin_areas = np.array([-2*np.pi*np.cos(bin_edges[i+1]) + 2*np.pi*np.cos(bin_edges[i]) for i in range(n_bins)])
        assert np.sum(bin_areas) - 4*np.pi < 1e-10
        expected_counts = n_tests * bin_areas / (4*np.pi)
    
        assert_chisquared(observed_counts, expected_counts, bin_edges, alpha=0.05, title='theta (polar angle)')
    
    yield assert_phi_chi2
    yield assert_theta_chi2


def assert_chisquared(observed, expected, bin_edges=None, alpha=0.05, title=''):
    """Assert that the "observed" counts are close enough the "expected"
    counts with a chi2 test at the given confidence level. If the test fails,
    we'll save a histogram to disk.
    """

    
    chi2_value = np.sum((observed - expected)**2 / expected)
    
    n_dof = len(observed) - 1 # number of degrees of freedom for the chi2 test
    pval = 1 - chi2distribution.cdf(chi2_value, n_dof)
    
    print 'observed'
    print observed
    print 'expected'
    print expected
    print 'pval', pval
    
    if pval < alpha:
        if HAVE_PYPLOT:
            pp.clf()
            pp.title(title)
            pp.bar(bin_edges[0:-1], observed, width=bin_edges[1] - bin_edges[0],
                   alpha=0.5, label='observed', color='red')
            pp.bar(bin_edges[0:-1], expected, width=bin_edges[1] - bin_edges[0],
                   alpha=0.5, label='expected', color='blue')
            pp.legend()
            for i in itertools.count():
                path = 'hist-%d.png' % i
                if not os.path.exists(path):
                    break
            pp.savefig(path)
        raise ValueError('p=%f (<%f), we reject null hypothesis that the distribution '
            'matches the expected one. saved histogram as %s' % (pval, alpha, path))
