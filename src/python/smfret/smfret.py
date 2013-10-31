
#from smfret import distances
from odin.smfret.distances import residue_residue, atom_atom
from odin.exptdata import SingleMolecExperiment
import numpy as np

class smFRET(SingleMolecExperiment):
    """
    Class for working with a single molecule FRET (Forster 
    Resonance Energy Transfer) experiment. This is a single
    molecule experiment, and so the data should be provided 
    as a histogram over FRET efficiencies.

    The histogram is then used as a multinomial distribution
    for estimating the probability of an observed histogram
    given the experimental measurement.
    
    Parameters
    ----------
    bins : np.ndarray
        all points to specify the bin boundaries of the FRET 
        efficiency data. This should correspond to the left-most
        and right-most boundaries! So the length of `bins` should 
        be one greater than the length of `heights`
    heights : np.ndarray
        heights for each bin. This should have one fewer entry
        than the bins, since the bins are defined by the intervals
        between the bin boundaries
    forster_radius : float
        forster_radius (R_0) for the given FRET acceptor-donor
        pair. This will change the predicted FRET observables
    distance_fcn : function or one of {'residue-residue', 'atom-atom'}
        This function provides the donor acceptor distance for
        determining the FRET efficiency. Since this can be done
        in a number of ways, there are several options, the first
        being a user defined function. There are two built-in
        functions corresponding to a residue-residue distance
        and an atom atom distance. 
    distance_kwargs : dict
        additional kwargs to pass to the distance_fcn. If you
        selected residue-residue, then you must pass "inds" which
        is a length two array corresponding to the two residue positions
        with the donor and acceptors. If you used atom-atom, you
        must pass "inds" corresponding to two atoms to approximate
        the donor-acceptor distance. In both cases, indices are
        zero-indexed.
    """

    _builtin_distance_fcns = {'residue-residue' : residue_residue,
                              'atom-atom' : atom_atom}

    @classmethod
    def load(cls, filename):
        """
        load a file corresponding to an experimental measurement
        """
        raise NotImplementedError("not sure how this should work in this case..")


    def __init__(self, bins, heights, forster_radius, 
        distance_fcn='residue-residue', distance_kwargs={}, **kwargs):
        
        self._bins = np.array(bins).flatten()
        self._heights = np.array(heights).flatten()

        if self._bins.shape[0] != (self._heights.shape[0] + 1):
            raise Exception("there should be one more bin entry "
                "than in heights...")
    
        distance_kwargs = dict(distance_kwargs.items() + kwargs.items())
        # grab the extra kwargs into this dictionary too. This is
        # just a convenience function so you can pass inds=[0, 1]
        # instead of distance_kwargs={'inds':[0, 1]}

        if distance_fcn in ['residue-residue', 'atom-atom']:
            if not 'inds' in distance_kwargs.keys():
                raise Exception("need to input `inds` to use the builtin " 
                    "distance function.")

            distance_fcn = self._builtin_distance_fcns[distance_fcn]

        self.get_donor_acceptor_distance = lambda traj : distance_fcn(traj, **distance_kwargs)

        self.forster_radius = forster_radius
        

    def predict(self, trajectory):
        """
        Predict the FRET efficiencies for each frame in `trajectory`
        
        Parameters
        ----------
        trajectory : mdtraj.Trajectory
            trajectory to get the FRET efficiencies for

        Returns
        -------
        predictions : np.ndarray
            predicted FRET efficiency for each conformation
        """

        distances = self.get_donor_acceptor_distance(trajectory)
        R6 = np.power(distances / self.forster_radius, 6)
        predictions = 1. / (1. + (R6))

        return predictions


    def log_likelihood(self, predictions, weights=None):
        """
        We want to know how likely this ensemble is given the experiment
        but we will use Bayes' rule to calculate the probability of
        observing the experimental histogram given the ensemble data. 
        The strategy will be to histogram the  distribution based on 
        the provided weights to fit a multinomial distribution.

        This multinomial is then used to evaluate the likelihood
        of the experimental histogram.
    
        Parameters
        ----------
        predictions : np.ndarray
            one-dimensional array corresponding to the FRET efficiency
            for each point in the structural ensemble
        weights : np.ndarray
            weight of each point in the ensemble
        
        Returns
        -------
        log_likelihood : float
            log likelihood of observing the experimental data given
            the multinomial parameterized by the structural ensemble
        """

        if weights is None:
            weights = np.ones(predictions.shape[0])

        weights = weights / weights.sum()

        heights, edges = np.histogram(predictions, bins=self._bins, weights=weights)

        bin_probs = heights / heights.sum()

        log_likelihood = np.log(bin_probs).dot(self._heights)
        # ^^ technically there is a prefactor too, which corresponds 
        # to the degenerate ways of getting self._heights, but since 
        # the prefactor does not change as a function of the weights,
        # this will be fine..
        return log_likelihood
