# THIS FILE IS PART OF ODIN

"""
Classes, methods, functions for use with xray scattering experiments.
"""

import logging
logging.basicConfig()
logger = logging.getLogger(__name__)
#logger.setLevel('DEBUG')

import os
import cPickle
import tables

import numpy as np
from scipy import interpolate
from scipy.ndimage import filters, interpolation
from scipy.special import legendre

from odin import math2
from odin import scatter
from odin import utils
from odin import parse

from mdtraj import io
from mdtraj.utils.arrays import ensure_type


# ------------------------------------------------------------------------------
# FUNDAMENTAL CONSTANTS

h = 4.135677516e-15   # Planks constant | eV s
c = 299792458         # speed of light  | m / s

# ------------------------------------------------------------------------------


class Beam(object):
    """
    Class that converts energies, wavelengths, frequencies, and wavenumbers.
    Each instance of this class can represent a light source.

    Attributes
    ----------
    self.energy      (keV)
    self.wavelength  (angstroms)
    self.frequency   (Hz)
    self.wavenumber  (angular, inv. angstroms)
    """

    def __init__(self, photons_scattered_per_shot, **kwargs):
        """
        Generate an instance of the Beam class.

        Parameters
        ----------
        photons_scattered_per_shot : int
            The average number of photons scattered per shot.

        **kwargs : dict
            Exactly one of the following, in the indicated units
            -- energy:     keV
            -- wavelength: angstroms
            -- frequency:  Hz
            -- wavenumber: inverse angstroms
        """

        self.photons_scattered_per_shot = photons_scattered_per_shot

        # make sure we have only one argument
        if len(kwargs) != 1:
            raise KeyError('Expected exactly one argument, got %d' % (len(args)+1) )

        self.units = 'energy: keV, wavelengths: angstroms, frequencies: Hz, wavenumbers: inverse angstroms'

        # no matter what gets provided, go straight to energy and then
        # convert to the rest from there
        for key in kwargs:

            if key == 'energy':
                self.energy = float(kwargs[key])

            elif key == 'wavenumber':
                self.wavenumber = float(kwargs[key])
                self.energy = self.wavenumber * h * c * 10.**7. / (2.0 * np.pi)

            elif key == 'wavelength':
                self.wavelength = float(kwargs[key])
                self.energy = h * c * 10.**7. / self.wavelength

            elif key == 'frequency':
                self.frequency = float(kwargs[key])
                self.energy = self.frequency * h

            else:
                raise ValueError('%s not a recognized kwarg' % key)

        # perform the rest of the conversions
        self.wavelength = h * c * 10.**7. / self.energy
        self.wavenumber = 2.0 * np.pi / self.wavelength
        self.frequency = self.energy * (1000. / h)

        # some aliases
        self.k = self.wavenumber


class BasisGrid(object):
    """
    A class representing a set of rectangular grids in space -- specifically,
    x-ray scattering detectors. Does not contain all the metadata associated
    with a full-fledged Detector class (e.g. the wavelength, etc).

    Note that the geometry below is definied in "slow" and "fast" scan
    dimensions. These are simply the two dimensions that define the plane
    a single rectangular pixel grid lives in. They may also be called the y and
    x dimensions without any loss of generality.

    The convention here -- and in all of ODIN -- is one of Row-Major ordering,
    which is consistent with C/python. This means that y is the slow dim, x is
    the fast dim, and when ordering these two they will appear as (slow, fast).

    Note on units: units are arbitrary -- all the units must be the same for
    this to work. We don't keep track of units here.

    The class is a set of rectangular grids, with each grid defined by four
    quantities:

        -- p vector : DEFINES A GRIDS POSITION IN SPACE.
                      The vector between a chosen origin (possibly interaction
                      site) and the corner of the grid that is smallest in both
                      slow and fast (x/y) dimensions of the coordinate system.
                      Usually this will be the "bottom left" corner, but due to
                      the generality of the coordinates used, this is not
                      necessarily true.

        -- s/f vect : DEFINES A GRIDS ORIENTATION IN SPACE
                      Vectors pointing along the slow/fast-scan direction,
                      respectively. These define the plane containing the pixels.
                      The magnitudes of these vectors defines the size of the
                      pixel in that dimension.

        -- shape    : DEFINES GRID DIMENSIONS
                      The number of pixels in the fast/slow direction. Ints.
    """


    def __init__(self, list_of_grids=[]):
        """
        Initialize a BasisGrid object.

        Parameters
        ----------
        list_of_grids : list
            A list of tuples of the form  (p, s, f, shape). See the doc
            for the `add_grid` method on this class for more information. May
            be an empty list (default) in which case a GridList with no pixels
            is created.

        See Also
        --------
        add_grid
        add_grid_using_center
        """

        if not type(list_of_grids) == list:
            raise TypeError('`list_of_grids` must be a list')

        self._num_grids = 0
        self._ps        = [] # p-vectors
        self._ss        = [] # slow-scan vectors
        self._fs        = [] # fast-scan vectors
        self._shapes    = [] # shapes

        if len(list_of_grids) > 0:
            for grid in list_of_grids:
                self.add_grid(*grid)

        return


    def _check_valid_basis(self, p, s, f, shape):
        """
        Check to make sure that all the inputs look good.
        """
            
        ensure_type(p, np.float, 1, 'p', length=3)
        ensure_type(s, np.float, 1, 's', length=3)
        ensure_type(f, np.float, 1, 'f', length=3)

        return


    def _assert_list_sizes(self):
        """
        A simple sanity check
        """
        assert len(self._ps)     == self.num_grids
        assert len(self._ss)     == self.num_grids
        assert len(self._fs)     == self.num_grids
        assert len(self._shapes) == self.num_grids
        return


    @property
    def num_pixels(self):
        """
        Return the total number of pixels in the BasisGrid.
        """
        n = np.sum([np.product(self._shapes[i]) for i in range(self.num_grids)])
        return int(n)


    @property
    def num_grids(self):
        return self._num_grids


    def add_grid(self, p, s, f, shape):
        """
        Add a grid (detector array) to the basis representation.

        Parameters
        ----------
        p : np.ndarray, float
            3-vector from the origin to the pixel on the grid with
            smallest coordinate in all dimensions.

        s : np.ndarray, float
            3-vector pointing in the slow scan direction

        f : np.ndarray, float
            3-vector pointing in the slow scan direction

        shape : tuple or list of float
            The number of pixels in the (slow, fast) directions. Len 2.

        See Also
        --------
        add_grid_using_center
        """
        self._check_valid_basis(p, s, f, shape)
        self._ps.append(p)
        self._ss.append(s)
        self._fs.append(f)
        self._shapes.append(shape)
        self._num_grids += 1
        self._assert_list_sizes()
        return


    def add_grid_using_center(self, p_center, s, f, shape):
        """
        Add a grid (detector array) to the basis representation. Here, though,
        the p-vector points to the center of the array instead of the slow/fast
        smallest corner.

        Parameters
        ----------
        p_center : np.ndarray, float
            3-vector from the origin to the center of the grid.

        s : np.ndarray, float
            3-vector pointing in the slow scan direction

        f : np.ndarray, float
            3-vector pointing in the slow scan direction

        shape : tuple or list of float
            The number of pixels in the (slow, fast) directions. Len 2.
        """

        p_center = np.array(p_center)
        if not p_center.shape == (3,):
            raise ValueError('`p_center` must have shape (3,)')

        # just compute where `p` is then add the grid as usual
        x = (np.array(shape) - 1)
        center_correction =  ((x[0] * s) + (x[1] * f)) / 2.
        p  = p_center.copy()
        p -= center_correction

        self.add_grid(p, s, f, shape)

        return


    def get_grid(self, grid_number):
        """
        Return a grid for grid `grid_number`.

        Parameters
        ----------
        grid_number : int
            The index of the grid to get.

        Returns
        -------
        p_center : np.ndarray, float
            3-vector from the origin to the center of the grid.

        s : np.ndarray, float
            3-vector pointing in the slow scan direction

        f : np.ndarray, float
            3-vector pointing in the slow scan direction

        shape : tuple or list of float
            The number of pixels in the (slow, fast) directions. Len 2.
        """

        if grid_number >= self.num_grids:
            raise ValueError('Only %d grids in object, you asked for the %d-th'
                             ' (zero indexed)' % (self.num_grids, grid_number))

        grid_tuple = (self._ps[grid_number], self._ss[grid_number],
                      self._fs[grid_number], self._shapes[grid_number])

        return grid_tuple


    def get_grid_corners(self, grid_number):
        """
        Return the positions of the four corners of a grid.

        Parameters
        ----------
        grid_number : int
            The index of the grid to get the corners of.

        Returns
        -------
        corners : np.ndarray, float
            A 4 x 3 array, where the first dim represents the four corners, and
            the second is x/y/z. Note one corner is always just the `p` vector.
        """

        if grid_number >= self.num_grids:
            raise ValueError('Only %d grids in object, you asked for the %d-th'
                             ' (zero indexed)' % (self.num_grids, grid_number))

        # compute the lengths of the parallelogram sides
        s_side = self._ss[grid_number] * float(self._shapes[grid_number][0])
        f_side = self._fs[grid_number] * float(self._shapes[grid_number][1])
        pc = self._ps[grid_number].copy()

        corners = np.zeros((4,3))

        corners[0,:] = pc
        corners[1,:] = pc + s_side
        corners[2,:] = pc + f_side
        corners[3,:] = pc + s_side + f_side

        return corners


    def to_explicit(self):
        """
        Return the entire grid as an n x 3 array, defining the x,y,z positions
        of each pixel.

        Returns
        -------
        xyz : np.ndarray, float
            An N x 3 array of the x,y,z positions of each pixel. Note that this
            is a flattened version of what you get for each grid individually
            using `grid_as_explicit`.

        See Also
        --------
        grid_as_explicit
        """
        ex_grids = [ self.grid_as_explicit(i) for i in range(self.num_grids) ]
        xyz = np.concatenate([ g.reshape((g.shape[0]* g.shape[1], 3)) for g in ex_grids ])
        return xyz


    def grid_as_explicit(self, grid_number):
        """
        Get the x,y,z coordiantes for a single grid.

        Parameters
        ----------
        grid_number : int
            The index of the grid to get.

        Returns
        -------
        xyz : np.ndarray, float
            An (shape) x 3 array of the x,y,z positions of each pixel

        See Also
        --------
        to_explicit
        """

        p, s, f, shape = self.get_grid(grid_number)

        # xyz = i * s + j * f, where i,j are ints running over range `shape`
        mg = np.mgrid[0:shape[0]-1:1j*shape[0], 0:shape[1]-1:1j*shape[1]]
        xyz = np.outer(mg[0].flatten(), s) + np.outer(mg[1].flatten(), f)
        xyz += p # translate
        xyz = xyz.reshape( (shape[0], shape[1], 3) )

        return xyz


class Detector(Beam):
    """
    Class that provides a plethora of geometric specifications for a detector
    setup. Also provides loading and saving of detector geometries.
    """

    def __init__(self, xyz, k, beam_vector=None):
        """
        Instantiate a Detector object.

        Detector objects provide a handle for the many representations of
        detector geometry in scattering experiments, namely:

        -- real space
        -- real space in polar coordinates
        -- reciprocal space (q-space)
        -- reciprocal space in polar coordinates (q, theta, phi)

        Note the the origin is assumed to be the interaction site.

        Parameters
        ----------
        xyz : ndarray OR xray.BasisGrid
            An a specification the (x,y,z) positions of each pixel. This can
            either be n x 3 array with the explicit positions of each pixel,
            or a BasisGrid object with a vectorized representation of the
            pixels. The latter yeilds higher performance, and is recommended.

        k : float or odin.xray.Beam
            The wavenumber of the incident beam to use. Optionally a Beam
            object, defining the beam energy.

        Optional Parameters
        -------------------
        beam_vector : float
            The 3-vector describing the beam direction. If `None`, then the
            beam is assumed to be purely in the z-direction.
        """

        if type(xyz) == np.ndarray:
            logger.debug('xyz type: np.ndarray, initializing an explicit detector')

            self._pixels = xyz
            self._basis_grid = None
            self.num_pixels = xyz.shape[0]
            self._xyz_type = 'explicit'

        elif type(xyz) == BasisGrid:
            logger.debug('xyz type: BasisGrid, initializing an implicit detector')

            self._pixels = None
            self._basis_grid = xyz
            self.num_pixels = self._basis_grid.num_pixels
            self._xyz_type = 'implicit'

        else:
            raise TypeError("`xyz` type must be one of {'np.ndarray', "
                            "'odin.xray.BasisGrid'}")


        # parse wavenumber
        if isinstance(k, Beam):
            self.k = k.wavenumber
            self.beam = k
        elif type(k) in [float, np.float64, np.float32]:
            self.k = k
            self.beam = None
        else:
            raise TypeError('`k` must be a float or odin.xray.Beam')

        # parse beam_vector -- is guarenteed to be a unit vector
        if beam_vector != None:
            if beam_vector.shape == (3,):
                self.beam_vector = self._unit_vector(beam_vector)
            else:
                raise ValueError('`beam_vector` must be a 3-vector')
        else:
            self.beam_vector = np.array([0.0, 0.0, 1.0])

        return


    def implicit_to_explicit(self):
        """
        Convert an implicit detector to an explicit one (where the xyz pixels
        are stored in memory).
        """
        if not self.xyz_type == 'implicit':
            raise Exception('Detector must have xyz_type implicit for conversion.')
        self._pixels = self.xyz
        self._xyz_type = 'explicit'
        return


    @property
    def xyz_type(self):
        return self._xyz_type


    @property
    def xyz(self):
        if self.xyz_type == 'explicit':
            return self._pixels
        elif self.xyz_type == 'implicit':
            return self._basis_grid.to_explicit()


    @property
    def real(self):
        return self.xyz.copy()


    @property
    def polar(self):
        return self._real_to_polar(self.real)


    @property
    def reciprocal(self):
        return self._real_to_reciprocal(self.real)


    @property
    def recpolar(self):
        a = self._real_to_recpolar(self.real)
        # convention: theta is angle of q-vec with plane normal to beam
        a[:,1] = self.polar[:,1] / 2.0
        return a


    @property
    def q_max(self):
        """
        Returns the maximum value of |q| the detector measures
        """

        if self.xyz_type == 'explicit':
            q_max = np.max(self.recpolar[:,0])

        elif self.xyz_type == 'implicit':
            q_max = 0.0
            for i in range(self._basis_grid.num_grids):
                c  = self._basis_grid.get_grid_corners(i)
                qc = self._real_to_recpolar(c)
                q_max = max([q_max, float(np.max(qc[:,0]))])

        return q_max


    def evaluate_qmag(self, xyz):
        """
        Given the positions of pixels `xyz`, compute the corresponding |q|
        value for each.

        Parameters
        ----------
        qxyz : ndarray, float
            The array of pixels (shape : N x 3)

        Returns
        -------
        qmag : ndarray, float
            The array of q-magnitudes, len N.
        """
        thetas = self._evaluate_theta(xyz)
        qmag = 2.0 * self.k * np.sin(thetas/2.0)
        return qmag


    def _evaluate_theta(self, xyz):
        """
        Given the positions of pixels `xyz`, compute the corresponding
        scattering angle theta for each.

        Parameters
        ----------
        xyz : ndarray, float
            The array of pixels (shape : N x 3)

        Returns
        -------
        thetas : ndarray, float
            The scattering angles for each pixel
        """
        u_xyz  = self._unit_vector(xyz)
        thetas = np.arccos(np.dot( u_xyz, self.beam_vector ))
        return thetas


    def _real_to_polar(self, xyz):
        """
        Convert the real-space representation to polar coordinates.
        """
        polar = self._to_polar(xyz)
        return polar


    def _real_to_reciprocal(self, xyz):
        """
        Convert the real-space to reciprocal-space in cartesian form.
        """

        assert len(xyz.shape) == 2
        assert xyz.shape[1] == 3

        # generate unit vectors in the pixel direction, origin at sample
        S = self._unit_vector(xyz)
        q = self.k * (S - self.beam_vector)

        return q


    def _real_to_recpolar(self, xyz):
        """
        Convert the real-space to reciprocal-space in polar form, that is
        (|q|, theta , phi).
        """
        reciprocal_polar = self._to_polar( self._real_to_reciprocal(xyz) )
        return reciprocal_polar


    @staticmethod
    def _norm(vector):
        """
        Compute the norm of an n x m array of vectors, where m is the dimension.
        """
        if len(vector.shape) == 2:
            assert vector.shape[1] == 3
            norm = np.sqrt( np.sum( np.power(vector, 2), axis=1 ) )
        elif len(vector.shape) == 1:
            assert vector.shape[0] == 3
            norm = np.sqrt( np.sum( np.power(vector, 2) ) )
        else:
            raise ValueError('Shape of vector wrong')
        return norm


    def _unit_vector(self, vector):
        """
        Returns a unit-norm version of `vector`.

        Parameters
        ----------
        vector : ndarray, float
            An n x m vector of floats, where m is assumed to be the dimension
            of the space.

        Returns
        -------
        unit_vectors : ndarray,float
            An n x m vector, same as before, but now of unit length
        """

        norm = self._norm(vector)

        if len(vector.shape) == 1:
            unit_vectors = vector / norm

        elif len(vector.shape) == 2:
            unit_vectors = np.zeros( vector.shape )
            for i in range(vector.shape[0]):
                unit_vectors[i,:] = vector[i,:] / norm[i]

        else:
            raise ValueError('invalid shape for `vector`: %s' % str(vector.shape))

        return unit_vectors


    def _to_polar(self, vector):
        """
        Converts n m-dimensional `vector`s to polar coordinates. By polar
        coordinates, I mean the cannonical physicist's (r, theta, phi), no
        2-theta business.

        We take, as convention, the 'z' direction to be along self.beam_vector
        """

        polar = np.zeros( vector.shape )

        # note the below is a little modified from the standard, to take into
        # account the fact that the beam may not be only in the z direction

        polar[:,0] = self._norm(vector)
        polar[:,1] = np.arccos( np.dot(vector, self.beam_vector) / \
                                (polar[:,0]+1e-16) )           # cos^{-1}(z.x/r)
        polar[:,2] = math2.arctan3(vector[:,1] - self.beam_vector[1],
                                   vector[:,0] - self.beam_vector[0])   # y first!

        return polar


    def _compute_intersections(self, q_vectors, grid_index, run_checks=True):
        """
        Compute the points i=(x,y,z) where the scattering vectors described by
        `q_vectors` intersect the detector.

        Parameters
        ----------
        q_vectors : np.ndarray
            An N x 3 array representing q-vectors in cartesian q-space.

        grid_index : int
            The index of the grid array to intersect

        Optional Parameters
        -------------------
        run_checks: bool
            Whether to run some good sanity checks, at small computational cost.

        Returns
        -------
        pix_n : ndarray, float
            The coefficients of the position of each intersection in terms of
            the basis grids s/f vectors.

        intersect: ndarray, bool
            A boolean array of which of `q_vectors` intersect with the grid
            plane. First column is slow scan vector (s),  second is fast (f).

        References
        ----------
        .[1] http://en.wikipedia.org/wiki/Line-plane_intersection
        """

        if not self.xyz_type == 'implicit':
            raise RuntimeError('intersections can only be computed for implicit'
                               ' detectors')

        # compute the scattering vectors correspoding to q_vectors
        S = (q_vectors / self.k) + self.beam_vector

        # compute intersections
        p, s, f, shape = self._basis_grid.get_grid(grid_index)
        n = self._unit_vector( np.cross(s, f) )
        i = (np.dot(p, n) / np.dot(S, n))[:,None] * S

        # convert to pixel units by solving for the coefficients of proj
        A = np.array([s,f]).T
        pix_n, resid, rank, sigma = np.linalg.lstsq( A, (i-p).T )

        if run_checks:
            err = np.sum( np.abs((i-p) - np.transpose( np.dot(A, pix_n) )) )
            if err > 1e-6:
                raise RuntimeError('Error in computing where scattering vectors '
                                   'intersect with detector. Intersect not reproduced'
                                   ' (err: %f per pixel)' % (err / i.shape[0],) )

            if not np.sum(resid) < 1e-6:
                raise RuntimeError('Error in basis grid (residuals of point '
                                   'placement too large). Perhaps fast/slow '
                                   'vectors describing basis grid are linearly '
                                   'dependant?')

        pix_n = pix_n.T
        assert pix_n.shape[1] == 2 # s/f

        # see if the intersection in the plane is on the detector grid
        intersect = (pix_n[:,0] >= 0.0) * (pix_n[:,0] <= float(shape[0]-1)) *\
                    (pix_n[:,1] >= 0.0) * (pix_n[:,1] <= float(shape[1]-1))

        logger.debug('%.3f %% of pixels intersected by grid %d' % \
            ( (float(np.sum(intersect)) / float(np.product(intersect.shape)) * 100.0),
            grid_index) )

        return pix_n[intersect], intersect


    @classmethod
    def generic(cls, spacing=1.00, lim=100.0, energy=10.0,
                photons_scattered_per_shot=1e4, l=50.0,
                force_explicit=False):
        """
        Generates a simple grid detector that can be used for testing
        (factory function).

        Optional Parameters
        -------------------
        spacing : float
            The real-space grid spacing
        lim : float
            The upper and lower limits of the grid
        energy : float
            Energy of the beam (in keV)
        l : float
            The path length from the sample to the detector, in the same units
            as the detector dimensions.
        force_explicit : bool
            Forces the detector to be xyz_type explicit. Mostly for debugging.
            Recommend keeping `False`.

        Returns
        -------
        detector : odin.xray.Detector
            An instance of the detector that meets the specifications of the
            parameters
        """

        beam = Beam(photons_scattered_per_shot, energy=energy)

        if not force_explicit:

            p = np.array([-lim, -lim, l])   # corner position
            f = np.array([0.0, spacing, 0.0]) # slow scan is x
            s = np.array([spacing, 0.0, 0.0]) # fast scan is y

            dim = int(2*(lim / spacing) + 1)
            shape = (dim, dim)

            basis = BasisGrid()
            basis.add_grid(p, s, f, shape)

            detector = cls(basis, beam)

        else:
            x = np.arange(-lim, lim+spacing, spacing)
            xx, yy = np.meshgrid(x, x)

            xyz = np.zeros((len(x)**2, 3))
            xyz[:,0] = yy.flatten() # fast scan is y
            xyz[:,1] = xx.flatten() # slow scan is x
            xyz[:,2] = l

            detector = cls(xyz, beam)

        return detector


    def _to_serial(self):
        """ serialize the object to an array """
        s = np.array( cPickle.dumps(self) )
        s.shape=(1,) # a bit nasty...
        return s


    @classmethod
    def _from_serial(self, serialized):
        """ recover a Detector object from a serialized array """
        if serialized.shape == (1,):
            serialized = serialized[0]
        d = cPickle.loads( str(serialized) )
        return d


    def save(self, filename):
        """
        Writes the current Detector to disk.

        Parameters
        ----------
        filename : str
            The path to the shotset file to save.
        """

        if not filename.endswith('.dtc'):
            filename += '.dtc'
            
        if os.path.exists(filename):
            raise IOError('File: %s already exists! Aborting...' % filename)

        io.saveh(filename, detector=self._to_serial())
        logger.info('Wrote %s to disk.' % filename)

        return


    @classmethod
    def load(cls, filename):
        """
        Loads the a Detector from disk.

        Parameters
        ----------
        filename : str
            The path to the shotset file.

        Returns
        -------
        shotset : odin.xray.Shotset
            A shotset object
        """

        if not filename.endswith('.dtc'):
            raise ValueError('Must load a detector file (.dtc extension)')

        hdf = io.loadh(filename)
        d = cls._from_serial(hdf['detector'])
        return d
    

class Shotset(object):
    """
    A collection of xray 'shots', and methods for anaylzing statistical
    properties across shots (each shot a single x-ray image).
        
    The key power/functionality of Shotset is that it provides a layer of
    abstraction over the intensity data on disk. Specifically, it provides many
    powerful capabilities for analyzing data sets that are larger than can
    fit in memory.
    """

    def __init__(self, intensities, detector, mask=None, filters=[]):
        """
        Instantiate a Shotset class.

        Parameters
        ----------
        intensities : ndarray, float OR tables.earray.EArray OR tables.carray.CArray
            There are two ways to pass intensity data: either as an explicit
            numpy array in memory, or as a tables array object that will iterate
            over individual shots.

        detector : odin.xray.Detector
            A detector object, containing the pixel positions in space.

        Optional Parameters
        -------------------
        mask : ndarray, np.bool
            An array the same size (and shape -- 1d) as `intensities` with a
            'np.True' in all indices that should be kept, and 'np.False'
            for all indices that should be masked.
            
        filters : list
            A list of callables that will get applied to the intensity data
            whenever it is accessed.
        """
        
        # initialize some internals
        self._intensity_filters = []
        if not utils.is_iterable(filters):
            raise TypeError('`filters` argument must be iterable')
        elif len(filters) > 0:
            for flt in filters:
                self._add_intensity_filter(flt)
        else:
            pass
        

        # parse detector
        if not isinstance(detector, Detector):
            raise ValueError('`detector` argument must be type: xray.Detector')
        else:
            self.detector = detector

        # parse intensities
        if type(intensities) == list:
            intensities = np.array(intensities)

        # this should *not* be an elif
        if type(intensities) in [np.ndarray, 
                                 tables.earray.EArray,
                                 tables.carray.CArray]:
            self._intensities = intensities
        else:
            raise TypeError('`intensities` must be type: {ndarray, EArray, '
                            'CArray}. Got %s' % type(intensities))


        # check that the data dimensions work out
        s = intensities.shape

        if len(s) == 1:
            if not s[0] == self.detector.num_pixels:
                raise ValueError('`intensities` does not have the same '
                                 'number of pixels as `detector`')
            self._intensities = intensities[None,:]

        elif len(s) == 2:
            if not s[1] == self.detector.num_pixels:
                raise ValueError('`intensities` does not have the same '
                                 'number of pixels as `detector`')
            self._intensities = intensities
            
        elif np.product(s[1:]) == self.detector.num_pixels:
            logger.debug('Non-linear intensity data :: attempting to filter')
            for itx in intensities:
                itx = self._filter_intensities(itx) 
                break
            if not itx.shape == (self.detector.num_pixels,):
                raise ValueError('`intensities` does not have the same '
                                 'number of pixels as `detector`')
                
        else:
            raise ValueError('`intensities` has a invalid number of '
                             'dimensions, must be 1 or 2 (got: %d)' % len(s))

        
        # parse mask
        if mask != None:
            mask = mask.flatten()
            if len(mask) != self.detector.num_pixels:
                raise ValueError('Mask must a len `detector.num_pixels` array')
            self.mask = np.array(mask.flatten()).astype(np.bool)
        else:
            self.mask = None


        return
    
        
    def close(self):
        """
        When class gets garbage collected, also close off all HDF5 handles.
        """
        if hasattr(self, '_file_handle'):
            if hasattr(self._file_handle, 'close'):
                logger.debug('Shotset.close :: closing file handle')
                self._file_handle.close()
        return
    
        
    @property
    def num_shots(self):
        """
        Return the number of shots. Note that this may be slow if the intensity
        data is large and on disk...
        """
        return int(self._intensities.shape[0])
    

    @property
    def num_pixels(self):
        return self.detector.num_pixels
    

    @property
    def intensities(self):
        """
        Fetch the intensity data from disk and return it as an array.
        """
        try:
            i_data = np.zeros((self.num_shots, self.num_pixels))
        except MemoryError as e:
            logger.critical(e)
            raise MemoryError('Insufficient memory to complete operation.'
                              ' Work with Shotset.intensities_iter().')
                              
        for i, itx in enumerate(self.intensities_iter):
            i_data[i,:] = itx
        
        return i_data
            
    
    @property        
    def intensities_iter(self):
        """
        An python generator providing a method to iterate over intensity data
        stored on disk.
        """
        if type(self._intensities) == np.ndarray:
            i_iter = self._intensities
        elif type(self._intensities) in [tables.earray.EArray,
                                         tables.carray.CArray]:
            i_iter = self._intensities.iterrows()
            i_iter.nrow = -1 # reset the iterator to the start
        else:
            raise RuntimeError('invalid type in self._intensities')
            
        # yield the filtered result
        for x in i_iter:
            yield self._filter_intensities(x)
        
        
    def _filter_intensities(self, intensities):
        """
        Apply any filters in self._intensity_filters to the intensity data
        and return the modified intensities.
        
        Parameters
        ----------
        intensities : np.ndarray
            The intensities to filter.
            
        Returns
        -------
        intensities : np.ndarray
            The filtered intensities
        """
        if len(self._intensity_filters) > 0:
            for flt in self._intensity_filters:
                intensities = flt(intensities)
        return intensities
    
        
    def _add_intensity_filter(self, flt):
        """
        Append a filter to the list of filters to apply to the intensity data.
        """
        if not hasattr(flt, '__call__'):
            raise TypeError('`flt` filter must be callable')
        else:
            self._intensity_filters.append(flt)
        return
    

    def __len__(self):
        return self.num_shots


    def __add__(self, other):
        
        raise NotImplementedError('addition not implemented')
        
        if not isinstance(other, Shotset):
            raise TypeError('Cannot add types: %s and Shotset' % type(other))
        if not self.detector == other.detector:
            raise RuntimeError('shotset objects must share the same detector to add')
        if not np.all(self.mask == other.mask):
            raise RuntimeError('shotset objects must share the same mask to add')
        new_i = np.vstack(( self.intensities, other.intensities ))
        return Shotset(new_i, self.detector, self.mask)


    @property
    def average_intensity(self):
        avg = np.zeros(self.num_pixels)
        for itx in self.intensities_iter:
            avg += itx
        avg /= float(self.num_shots)
        return avg
    

    @staticmethod
    def num_phi_to_values(num_phi):
        """
        Converts `phi_spacing` to the explicit values, all in RADIANS.
        """
        phi_values = np.arange(0, 2.0*np.pi, 2.0*np.pi/float(num_phi))
        return phi_values


    @staticmethod
    def num_phi_to_spacing(num_phi):
        return 2.0*np.pi / float(num_phi)


    def assemble_image(self, shot_index=None, num_x=None, num_y=None):
        """
        Assembles the Shot object into a real-space image.

        Parameters
        ----------
        shot_index : int
            The shot inside the Shotset to assemble. If `None`, will assemble
            an average image.

        num_x,num_y : int
            The number of pixels in the x/y direction that will comprise the final
            grid.

        Returns
        -------
        grid_z : ndarray, float
            A 2d array representing the image one would see when viewing the
            shot in real space. E.g., to visualize w/matplotlib:

            >>> imshow(grid_z.T)
            >>> show()
            ...
        """

        if shot_index == None:
            inten = self.average_intensity
        else:
            raise NotImplementedError()

        if (num_x == None) or (num_y == None):
            # todo : better performance if needed (implicit detector)
            num_x = len(self.detector.xyz[:,0])
            num_y = len(self.detector.xyz[:,1])

        points = self.detector.xyz[:,:2] # ignore z-comp. of detector

        x = np.linspace(points[:,0].min(), points[:,0].max(), num_x)
        y = np.linspace(points[:,1].min(), points[:,1].max(), num_y)
        grid_x, grid_y = np.meshgrid(x,y)

        grid_z = interpolate.griddata(points, inten,
                                      (grid_x,grid_y), method='nearest',
                                      fill_value=0.0)

        return grid_z


    def polar_grid(self, q_values, num_phi):
        """
        Return the pixels that comprise the polar grid in (q, phi) space.
        """

        phi_values = self.num_phi_to_values(num_phi)
        num_q = len(q_values)

        polar_grid = np.zeros((num_q * num_phi, 2))
        polar_grid[:,0] = np.repeat(q_values, num_phi)
        polar_grid[:,1] = np.tile(phi_values, num_q)

        return polar_grid


    def polar_grid_as_cart(self, q_values, num_phi):
        """
        Returns the pixels that comprise the polar grid in q-cartesian space,
        (q_x, q_y, q_z)
        """

        phi_values = self.num_phi_to_values(num_phi)
        num_q = len(q_values)

        pg_real = np.zeros((num_q * num_phi, 3))
        phis = np.tile(phi_values, num_q)
        pg_real[:,0] = np.repeat(q_values, num_phi) * np.cos( phis )
        pg_real[:,1] = np.repeat(q_values, num_phi) * np.sin( phis )

        return pg_real


    def interpolate_to_polar(self, q_values, num_phi, polar_intensities_output=None):
        """
        Interpolate our cartesian-based measurements into a polar coordiante
        system.

        Parameters
        ----------
        q_values : ndarray OR list OF floats
            If supplied, the interpolation will only be performed at these
            values of |q|, and the `q_spacing` parameter will be ignored.

        num_phi : int
            The number of equally-spaced points around the azimuth to
            interpolate (e.g. `num_phi`=360 means points at 1 deg spacing).
            
        Optional Parameters
        -------------------
        polar_intensities_output : object
            A variable output space for the interpolated intensities. If this
            is `None`, then `interpolated_intensities` will be returned as an
            array, as described below. Instead, you can pass any object to
            `polar_intensities_output` that implements an "append" method, and
            the object will be populated via that append method. A minimal
            example is polar_intensities_output=list, where list=[] which would
            cause the object "list" to be populated with the polar intensities.
            Mostly useful for writing stuff directly to disk.

        Returns
        -------
        interpolated_intensities : ndarray, float
            The interpolated values. A three-D array, (shots, q_values, phis)

        polar_mask : ndarray, bool
            A mask of ones and zeros. Ones are kept, zeros masked. Shape and
            pixels correspond to `interpolated_intensities`.
        """
        
        if polar_intensities_output == None:
            polar_intensities_output = []
        else:
            append = getattr(polar_intensities_output, "append", None)
            if not callable(append):
                raise RuntimeError('if `polar_intensities_output` is not None,'
                                   ' then it must implement an `append` method '
                                   'that can accept two-D np arrays in a sane'
                                   ' manner')

        # check to see what method we want to use to interpolate. Here,
        # `unstructured` is more general, but slower; implicit/structured assume
        # the detectors are grids, and are therefore faster but specific
        
        # `polar_intensities_output` is modified in-place

        if self.detector.xyz_type == 'explicit':
            polar_mask = self._explicit_interpolation(q_values,
                                                      num_phi, 
                                                      polar_intensities_output)
        elif self.detector.xyz_type == 'implicit':
            polar_mask = self._implicit_interpolation(q_values, 
                                                      num_phi, 
                                                      polar_intensities_output)
        else:
            raise RuntimeError('Invalid detector passed to Shot(), must be of '
                               'xyz_type {explicit, implicit}')

        if type(polar_intensities_output) == list:
            polar_intensities_output = np.vstack(polar_intensities_output)

        # polar_intensities_output "is" interpolated_intensities
        return polar_intensities_output, polar_mask


    def _implicit_interpolation(self, q_values, num_phi, polar_intensities_output):
        """
        Interpolate onto a polar grid from an `implicit` detector geometry.

        This detector geometry is specified by the x and y pixel spacing,
        the number of pixels in the x/y direction, and the top-left corner
        position.

        Notes
        -----
        --  The interpolation is performed in real space in basis vector (s/f)
            units
        --  The returned polar intensities are flattened, but each grid has data
            laid out as (q_values [slow], phi [fast])
            
        Parameters
        ----------
        q_values : ndarray OR list OF floats
            If supplied, the interpolation will only be performed at these
            values of |q|, and the `q_spacing` parameter will be ignored.

        num_phi : int
            The number of equally-spaced points around the azimuth to
            interpolate (e.g. `num_phi`=360 means points at 1 deg spacing).
        """

        # initialize output space for the polar data and mask
        num_q = len(q_values)
        polar_mask = np.zeros((num_q * num_phi), dtype=np.bool) # reshaped later
        q_vectors  = _q_grid_as_xyz(q_values, num_phi, self.detector.k)

        # these will store the:
        intersections = [] # (1) real-to-polar space map
        pix_ns = []        # coordinates where polar px intersect the real grid

        # --- loop over all the arrays that comprise the detector
        #     pre-compute where these intersect the detector
        #     also construct the polar mask

        int_start  = 0 # start of intensity array correpsonding to `grid`
        int_end    = 0 # end of intensity array correpsonding to `grid`                        

        for g in range(self.detector._basis_grid.num_grids):

            p, s, f, size = self.detector._basis_grid.get_grid(g)

            # compute how many pixels this grid has
            n_int = int( np.product(size) )
            int_end += n_int
            
            assert not int_end > self.num_pixels

            # compute where the scattering vectors intersect the detector
            pix_n, intersect = self.detector._compute_intersections(q_vectors, g)
            intersections.append(intersect)
            pix_ns.append(pix_n)

            if np.sum(intersect) == 0:
                logger.debug('Detector array (%d) had no pixels inside the '
                'interpolation area!' % g)
                int_start += n_int # increment
                continue
                
            # next, mask any points that should be masked by the real mask
            if self.mask == None:
                # unmask points that we have data for
                polar_mask[intersect] = np.bool(True)
            else:

                # to get the bicubic interpolation right, need to mask 16-px box
                # around any masked pixel. To do this, loop over masked px and
                # mask any polar pixel within 2-pixel units in either x or y dim

                assert self.mask.dtype == np.bool
                sub_mask = self.mask[int_start:int_end].reshape(size)
                sixteen_mask = filters.minimum_filter(sub_mask, size=(4,4),
                                                      mode='nearest')
                assert sixteen_mask.dtype == np.bool

                # copy the mask values from in sixteen_mask -- which is expanded
                # from the original mask to include an area of 16 px around each
                # originally masked px -- into the polar mask. False is masked.
                mp = np.floor(pix_n).astype(np.int)
                polar_mask[intersect] = sixteen_mask[mp[:,0],mp[:,1]]

            # increment index for self.intensities -- the real/measured intst.
            int_start += n_int
            
        assert len(intersections) == self.detector._basis_grid.num_grids
        polar_mask = polar_mask.reshape(num_q, num_phi)


        # --- loop over shots
        #     actually do the interpolation

        for shot,intensities in enumerate(self.intensities_iter):
            
            int_start  = 0 # start of intensity array correpsonding to `grid`
            int_end    = 0 # end of intensity array correpsonding to `grid`
            
            logger.info('interpolating shot %d/%d' % (shot+1, self.num_shots))
            shot_pi = np.zeros(num_q * num_phi)
            
            for g in range(self.detector._basis_grid.num_grids):
                
                intersect = intersections[g]
                pix_n     = pix_ns[g]
                
                # again, determine which px map to which grid
                p, s, f, size = self.detector._basis_grid.get_grid(g)
                n_int = int( np.product(size) )
                int_end += n_int
                assert not int_end > self.num_pixels
                
                if np.sum(intersect) > 0:

                    # interpolate onto the polar grid & update the inverse mask
                    # --> perform the interpolation in pixel units, and then convert
                    #     evaluated values to pixel units before evalutating
                    # employ scipy's bicubic interpolator, "map_coordinates", which
                    # takes a rectangular grid of intensities (sqI) and an array of 
                    # points to interpolate onto (pix_n)
                    
                    sqI = intensities[int_start:int_end].reshape(size[0], size[1])
                    shot_pi[intersect] = interpolation.map_coordinates(sqI, pix_n.T, 
                                                                       order=3,
                                                                       mode='nearest')
            
                int_start += n_int
            
            polar_intensities_output.append( shot_pi.reshape(1, num_q, num_phi) )
            
            
        return polar_mask


    def _explicit_interpolation(self, q_values, num_phi, polar_intensities_output):
        """
        Perform an interpolation to polar coordinates assuming that the detector
        pixels do not form a rectangular grid.

        NOTE: The interpolation is performed in polar momentum (q) space.
        """

        # initialize output space for the polar data and mask
        num_q = len(q_values)
        polar_mask = np.zeros(num_q * num_phi, dtype=np.bool)
        xy = self.detector.recpolar[:,[0,2]]

        # because we're using a "square" interplation method, wrap around one
        # set of polar coordinates to capture the periodic nature of polar coords
        add = ( xy[:,1] == xy[:,1].min() )
        xy_add = xy[add]
        xy_add[:,1] += 2.0 * np.pi
        aug_xy = np.concatenate(( xy, xy_add ))

        if self.mask != None:
            aug_mask = np.concatenate(( self.mask, self.mask[add] ))
        else:
            # slice all
            aug_mask = slice(0, self.detector.num_pixels + len(add))

        for intensities in self.intensities_iter:

            aug_int = np.concatenate(( intensities[:],
                                       intensities[add] ))

            # do the interpolation
            z_interp = interpolate.griddata( aug_xy[aug_mask], aug_int[aug_mask],
                                             self.polar_grid(q_values, num_phi),
                                             method='linear', fill_value=np.nan)
                                             
            # mask missing pixels (outside convex hull)
            nans = np.isnan(z_interp)
            z_interp[nans] = 0.0
            polar_mask[np.logical_not(nans)] = np.bool(True)

            # append this shot to the output
            polar_intensities_output.append( z_interp.reshape(1, num_q, num_phi) )

        polar_mask = polar_mask.reshape(num_q, num_phi)

        return polar_mask


    def intensity_profile(self, q_spacing=0.05):
        """
        Averages over the azimuth phi to obtain an intensity profile.

        Optional Parameters
        -------------------
        q_spacing : float
            The resolution of the |q|-axis, in inverse angstroms.

        Returns
        -------
        intensity_profile : ndarray, float
            An n x 2 array, where the first dimension is the magnitude |q| and
            the second is the average intensity at that point < I(|q|) >_phi.
        """

        q = self.detector.recpolar[:,0]
        q_vals = np.arange(q_spacing, q.max(), q_spacing)

        ind = np.digitize(q, q_vals)

        average_intensity = self.average_intensity
        avg = np.zeros(len(q_vals))
        for i in range(ind.max()):
            x = (ind == i)
            if np.sum(x) > 0:
                avg[i] = np.mean(average_intensity[x])
            else:
                avg[i] = 0.0

        intensity_profile = np.vstack( (q_vals, avg) ).T

        return intensity_profile


    def intensity_maxima(self, smooth_strength=10.0):
        """
        Find the positions where the intensity profile is maximized.

        Parameters
        ----------
        smooth_strength : float
            Controls the strength of the smoothing function used to make sure
            that noise is not picked up as a local maxima. Increase this value
            if you're picking up noise as maxima, decrease it if you're missing
            maxima.

        Returns
        -------
        maxima : list of floats
            A list of the |q| positions where the intensity is maximized.
        """

        # first, smooth; then, find local maxima based on neighbors
        intensity = self.intensity_profile()
        m = utils.maxima( math2.smooth(intensity[:,1], beta=smooth_strength) )
        return m


    @classmethod
    def simulate(cls, traj, detector, num_molecules, num_shots, traj_weights=None,
                 finite_photon=False, force_no_gpu=False, device_id=0):
        """
        Simulate a scattering 'shot', i.e. one exposure of x-rays to a sample, and
        return that as a Shot object (factory function).

        Assumes we have a Boltzmann distribution of `num_molecules` identical
        molecules (`trajectory`), exposed to a beam defined by `beam` and projected
        onto `detector`.

        Each conformation is randomly rotated before the scattering simulation is
        performed. Atomic form factors from X, finite photon statistics, and the
        dilute-sample (no scattering interference from adjacent molecules)
        approximation are employed.

        Parameters
        ----------
        traj : mdtraj.trajectory
            A trajectory object that contains a set of structures, representing
            the Boltzmann ensemble of the sample. If len(traj) == 1, then we assume
            the sample consists of a single homogenous structure, replecated
            `num_molecules` times.

        detector : odin.xray.Detector
            A detector object the shot will be projected onto.

        num_molecules : int
            The number of molecules estimated to be in the `beam`'s focus.

        num_shots : int
            The number of shots to simulate.

        Optional Parameters
        -------------------
        traj_weights : ndarray, float
            If `traj` contains many structures, an array that provides the Boltzmann
            weight of each structure. Default: if traj_weights == None, weights
            each structure equally.

        finite_photon : bool
            Use finite photon statistics in the simulation.

        force_no_gpu : bool
            Run the (slow) CPU version of this function.

        device_id : int
            The index of the GPU to run on.

        Returns
        -------
        shotset : odin.xray.Shotset
            A Shotset instance, containing the simulated shots.
        """

        I = np.zeros((num_shots, detector.num_pixels))

        for i in range(num_shots):
            I[i,:] = scatter.simulate_shot(traj, num_molecules, detector,
                                           traj_weights=traj_weights,
                                           finite_photon=finite_photon,
                                           force_no_gpu=force_no_gpu,
                                           device_id=device_id)

        ss = cls(I, detector)

        return ss


    def to_rings(self, q_values, num_phi=360, rings_filename=None):
        """
        Convert the shot to an xray.Rings object, for computing correlation
        functions and other properties in polar space.

        This automatically interpolates the dataset onto a polar grid and then
        converts those polar values into a class that facilitates computation
        in that space. See odin.xray.Rings for more info.

        Parameters
        ----------
        q_values : ndarray/list, float
            The values of |q| to extract rings at (in Ang^{-1}).

        num_phi : int
            The number of equally spaced points around the azimuth to
            interpolate onto (e.g. `num_phi`=360 means 1 deg spacing).
            
        rings_filename : str OR None
            A path to a file to write the rings object to. If `None` is
            provided, returns the rings object in memory. Writing to disk
            will alleviate memory use by storing the resulting rings object
            to disk in an intelligent way.
        """

        logger.info('Converting shotset to polar space (Rings)')
        
        try:
            q_values = np.array(q_values)
            assert len(q_values.shape) == 1
        except:
            raise TypeError('`q_values` must be a one-d array or list of floats')
                              
        # the easy way : keep everything in memory
        if not rings_filename:
            
            # polar_intensities_output=None automatically generates an array for
            # the output downstream
            pi, pm = self.interpolate_to_polar(q_values, num_phi)
            rings_obj = Rings(q_values, pi, self.detector.k, polar_mask=pm)
            ret_val = rings_obj
            
            
        # the good way : lazy load/write in chunks
        elif type(rings_filename) == str:
            
            if not rings_filename.endswith('.ring'):
                rings_filename += '.str'
                
            if os.path.exists(rings_filename):
                raise IOError('File with name %s already exists! Aborting...' \
                              % rings_filename)
                
            # generate the rings file on disk
            h5_handle = tables.File(rings_filename, 'w')
            
            # we want `polar_intensities` to be an EArray so we can add to it
            a = tables.Atom.from_dtype(np.dtype(np.float64))
            pi_node = h5_handle.createEArray(where='/', name='polar_intensities',
                                             shape=(0, len(q_values), num_phi), 
                                             atom=a, filters=io.COMPRESSION,
                                             expectedrows=self.num_shots)
                                            
            # populate the array on disk with the interpolated values
            pi, pm = self.interpolate_to_polar(q_values, num_phi, 
                                               polar_intensities_output=pi_node)
                                           
            # add metadata to the rings file
            io.saveh( h5_handle,
                      q_values = q_values,
                      k = np.array([self.detector.k]),
                      polar_mask = pm )

            logger.info('Wrote %s to disk.' % rings_filename)
            h5_handle.close()
            
            ret_val = None
            
            
        else:
            raise ValueError('`rings_filename` must be {None, str}')
            

        return ret_val


    def save(self, filename):
        """
        Writes the current Shotset data to disk.

        Parameters
        ----------
        filename : str
            The path to the shotset file to save.
        """

        if not filename.endswith('.shot'):
            filename += '.shot'
            
        if os.path.exists(filename):
            raise IOError('File: %s already exists! Aborting...' % filename)

        # if we don't have a mask, just save a single zero
        if self.mask == None:
            mask = np.array([0])
        else:
            mask = self.mask

        # save an inital file with all metadata
        hdf = tables.File(filename, 'w')
        
        io.saveh(hdf,
                 num_shots = np.array([self.num_shots]),
                 detector  = self.detector._to_serial(),
                 mask      = mask)
                          
        # add the intensity data bit by bit so we dont bloat memory
        a = tables.Atom.from_dtype(np.dtype(np.float64))
        pi_node = hdf.createEArray(where='/', name='intensities',
                                   shape=(0, self.num_pixels), 
                                   atom=a, filters=io.COMPRESSION,
                                   expectedrows=self.num_shots)
                                        
        for intx in self.intensities_iter:
            pi_node.append(intx[None,:])

        logger.info('Wrote %s to disk.' % filename)

        return


    @classmethod
    def load(cls, filename, force_into_memory=False, to_load=None):
        """
        Loads the a Shotset from disk. Must be `.shot` format.

        Parameters
        ----------
        filename : str
            The path to the shotset file.
            
        Optional Parameters
        -------------------
        force_into_memory : bool
            If `False`, shotset will attempt to keep as much data on disk as
            possible, and make passes over that data when necessary. This may
            make operations slightly less responsive, but lets you work with
            big data! If `True`, attempt to load everything into memory upfront.
            
        to_load : ndarray/list, ints
            The indices of the shots in `filename` to load. Can be used to sub-
            sample the shotset. Note this is not a valid option if
            `force_into_memory` is False. If `None`, load all data.

        Returns
        -------
        shotset : odin.xray.Shotset
            A shotset object
            
        See Also
        --------
        load_cxi : classmethod
            Load a CXIdb file as a shotset.
        """
        
        # figure out which shots to load
        if not to_load == None:
            try:
                to_load = np.array(to_load)
                assert to_load.dtype == np.int
            except:
                raise TypeError('`to_load` must be a ndarry/list of ints')


        # load from a shot file
        if not filename.endswith('.shot'):
            raise IOError('Invalid format for ShotSet file, must be .cxi, got:'
                          ' %s' % filename)
            
        hdf = tables.File(filename, 'r+')
        
        num_shots = int(hdf.root.num_shots.read())
        d = Detector._from_serial(hdf.root.detector.read())
        mask = hdf.root.mask.read()

        # check for our flag that there is no mask
        if np.all(mask == np.array([0])):
            mask = None

        if to_load == None: # load all
            if force_into_memory:
                intensities_handle = hdf.root.intensities.read()
                hdf.close()
                hdf = None
            else:
                intensities_handle = hdf.root.intensities
                
        else: # load subset
        
            if to_load.max() > num_shots + 1:
                raise ValueError('Asked to load shot %d in a data set of %d'
                                 ' total shots' % (to_load.max(), num_shots))
                
            if not force_into_memory:
                logger.warning('Shotset.load() recieved `to_load` flag while'
                               ' `force_into_memory` is False -- must load'
                               ' all data into memory to select subset. '
                               'Loading data into memory...')
                
            intensities_handle = np.zeros((len(to_load), d.num_pixels))
            logger.info('loading %d of %d shots...' % (len(to_load), num_shots))
            
            for i,s in enumerate(to_load):
                intensities_handle[i,:] = hdf.root.intensities.read(s)
            hdf.close()


        ss = cls(intensities_handle, d, mask)
        ss._file_handle = hdf

        return ss
    

    @classmethod
    def load_cxi(cls, filename, detector, mask):
        """
        Load a CXIdb file as a ShotSet.
        
        Parameters
        ----------
        filename : str
            Path to the CXIdb file on disk.
            
        detector : xray.Detector OR str
            Either a path to a .dtc file, or a detector object.
            
        mask : np.ndarray, bool OR str OR None
            A boolean mask to apply to the data. Can also be a pypad .mask file.
            Finally, can be 'None', in which case no mask is used.
        """
        
        if not filename.endswith('.cxi'):
            raise IOError('`load_cxi` can only load files with extension .cxi,'
            ' got: %s' % filename)
            
        if type(detector) == str:
            dtc = Detector.load(detector)
        elif isinstance(detector, Detector):
            dtc = detector
        else:
            raise TypeError('`detector` must be type {str, xray.Detector}, '
                            'got: %s' % (type(detector),))
                            
        if type(mask) == str:
            try:
                from pypad.mask import PadMask
            except ImportError as e:
                logger.critical(e)
                raise ImportError('must have pypad installed to load a .mask file')
            padmask = PadMask.load(mask)
            m = parse.CheetahCXI.cheetah_instensities_to_odin(padmask.mask2d)
        elif type(mask) == np.ndarray:
            m = mask.astype(np.bool)
        elif mask == None:
            m = None
        else:
            raise TypeError('`mask` must be type {str, np.ndarray}, '
                            'got: %s' % (type(mask),))
                            
        cxi = parse.CheetahCXI(filename)               
        ss = cls(cxi._ds1_data, dtc, m, 
                filters=[parse.CheetahCXI.cheetah_instensities_to_odin])
                
        # save a handle to the file so it doesn't get garbage collected
        ss._file_handle = cxi
        
        return ss


class Rings(object):
    """
    Class to keep track of intensity data in a polar space.
    """

    def __init__(self, q_values, polar_intensities, k, polar_mask=None,
                 filters=[]):
        """
        Interpolate our cartesian-based measurements into a polar coordiante
        system.

        Parameters
        ----------
        q_values : ndarray OR list OF floats
            The values of |q| in `polar_intensities`, in inverse Angstroms.

        polar_intensities : ndarray, float
            Intensities in polar space. Should be shape:

                N x len(`q_values`) x num_phi

            with N the number of shots (any value) and `num_phi` the number of
            points (equally spaced) around the azimuth.

        k : float
            The wavenumber of the energy used to acquire the data.

        Optional Parameters
        -------------------
        polar_mask : ndarray, bool
            A mask of ones and zeros. Ones are kept, zeros masked. Should be the
            same shape as `polar_intensities`, but LESS THE FRIST DIMENSION.
            That is, the polar mask is the same for all shots. Can also be
            `None`, meaning no masked pixels
            
        filters : list
            A list of callables that will get applied to the intensity data
            whenever it is accessed.
        """
        
        # initialize some internals
        self._intensity_filters = []
        if not utils.is_iterable(filters):
            raise TypeError('`filters` argument must be iterable')
        elif len(filters) > 0:
            for flt in filters:
                self._add_intensity_filter(flt)
        else:
            pass
            

        # this should *not* be an elif
        if type(polar_intensities) == np.ndarray:
            self._polar_intensities = np.copy( polar_intensities )
        elif type(polar_intensities) in [tables.earray.EArray,
                                         tables.carray.CArray]:
            self._polar_intensities = polar_intensities
        else:
            raise TypeError('`polar_intensities` must have type {ndarray, '
                            'EArray, CArray}. Got: %s' % type(polar_intensities))


        if not polar_intensities.shape[1] == len(q_values):
            raise ValueError('`polar_intensities` must have same len as '
                             '`q_values` in its second dimension.')

        if polar_mask == None:
            self.polar_mask = None
        elif type(polar_mask) == np.ndarray:
            if not polar_mask.shape == polar_intensities.shape[1:]:
                raise ValueError('`polar_mask` must have same shape as '
                                 '`polar_intensities[0,:,:]`,')
            if not polar_mask.dtype == np.bool:
                self.polar_mask = polar_mask.astype(np.bool)
            else:
                self.polar_mask = polar_mask
        else:
            raise TypeError('`polar_mask` must be np.ndarray or None')

        self._q_values = np.array(q_values)  # q values of the ring data
        self.k         = k                   # wave number

        return
    
        
    def close(self):
        if hasattr(self, '_hdf'):
            if self._hdf:
                self._hdf.close()
        return
    
        
    @property
    def _polar_intensities_type(self):
        if type(self._polar_intensities) == np.ndarray:
            type_str = 'array'
        elif type(self._polar_intensities) in [tables.earray.EArray,
                                               tables.carray.CArray]:
            type_str = 'tables'
        else:
            raise RuntimeError('incorrect type in self._intensities')
        return type_str


    @property
    def polar_intensities(self):
        try:
            i_data = np.zeros((self.num_shots, self.num_q, self.num_phi))
        except MemoryError as e:
            logger.critical(e)
            raise MemoryError('Insufficient memory to complete operation.'
                              ' Work with Shotset.intensities_iter().')
                              
        for i, itx in enumerate(self.polar_intensities_iter):
            i_data[i,:] = itx
        
        return i_data
    
        
    @property
    def polar_intensities_iter(self):
        if self._polar_intensities_type == 'array':
            pi_iter = self._polar_intensities
        elif self._polar_intensities_type == 'tables':
            pi_iter = self._polar_intensities.iterrows()
            pi_iter.nrow = -1 # reset the iterator to the start
            
        # yield the filtered result
        for x in pi_iter:
            yield self._filter_intensities(x)
            
            
    def _filter_intensities(self, intensities):
        """
        Apply any filters in self._intensity_filters to the intensity data
        and return the modified intensities.

        Parameters
        ----------
        intensities : np.ndarray
            The intensities to filter.

        Returns
        -------
        intensities : np.ndarray
            The filtered intensities
        """
        if len(self._intensity_filters) > 0:
            for flt in self._intensity_filters:
                intensities = flt(intensities)
        return intensities


    def _add_intensity_filter(self, flt):
        """
        Append a filter to the list of filters to apply to the intensity data.
        """
        if not hasattr(flt, '__call__'):
            raise TypeError('`flt` filter must be callable')
        else:
            self._intensity_filters.append(flt)
        return
    

    @property
    def num_shots(self):
        return self._polar_intensities.shape[0]


    @property
    def phi_values(self):
        return np.arange(0, 2.0*np.pi, 2.0*np.pi/float(self.num_phi))


    @property
    def q_values(self):
        return self._q_values


    @property
    def num_phi(self):
        return self._polar_intensities.shape[2]


    @property
    def num_q(self):
        return len(self._q_values)


    @property
    def num_datapoints(self):
        return self.num_phi * self.num_q
    
        
    def cospsi(self, q1, q2):
        """
        For each value if phi, compute the cosine of the angle between the
        reciprocal scattering vectors q1/q2 at angular separation phi.
        
        Parameters
        ----------
        q1/q2 : float
            The |q| values, in inv. ang.
            
        Returns
        -------
        cospsi : ndarray, float
            The cosine of psi, the angle between the scattering vectors.
        """
        
        t1     = np.pi/2. + np.arcsin( q1 / (2.*self.k) ) # theta 1 in spherical coor
        t2     = np.pi/2. + np.arcsin( q2 / (2.*self.k) ) # theta 2 in spherical coor
        cospsi = np.cos(t1)*np.cos(t2) + np.sin(t1)*np.sin(t2) *\
                 np.cos( self.phi_values )
              
        return cospsi
    

    def q_index(self, q, tolerance=1e-4):
        """
        Convert value of |q| (in inverse Angstroms) into the index used to
        slice `polar_intensities`.

        Parameters
        ----------
        q : float
            The value of |q| in inv Angstroms

        tolerance : float
            The tolerance in |q|. Will return values of q that are within this
            tolerance.

        Returns
        -------
        q_ind : int
            The index to slice `polar_intensities` at to get |q|.
        """
	    
        # check if there are rings at q
        q_ind = np.where( np.abs(self.q_values - q) < tolerance )[0]
        
        if len(q_ind) == 0:
            raise ValueError("No ring data at q="+str(q) +" inv ang. " \
                             "There are only data for q="+", ".join(np.char.mod("%.2f",self.q_values) )  )
        elif len(q_ind) > 1:
            raise ValueError("Multiple q-values found! Try decreasing the value"
                             "of the `tolerance` parameter.")
        
        return int(q_ind)


    def depolarize(self, xaxis_polarization):
        """
        Applies a polarization correction to the rings.
        
        Parameters
        ----------
        xaxis_polarization : float
            The fraction of the beam polarization in the horizontal/x plane.
            For synchrotron sources, this is the ''in-plane'' polarization.
            
        Citations
        ---------
        ..[1] Hura et. al. J. Chem. Phys. 113, 9140 (2000); doi10.1063/1.1319614
        ..[2] Jackson. Classical Electrostatics.
        """
        
        logger.info('Applying polarization correction w/P_x=%.3f' % xaxis_polarization)
        if (xaxis_polarization > 1.0) or (xaxis_polarization < 0.0):
            raise ValueError('Polarization cannot be greater than 100%! Got '
                             '`xaxis_polarization` value of %.3f' % xaxis_polarization)

        correctn = np.zeros((self.num_q, self.num_phi))

        for i,q in enumerate(self.q_values):
            theta     = np.arcsin( q / (2.0 * self.k) )
            sin_theta = np.sin(2.0 * theta)
            correctn[i,:]  = (1.-xaxis_polarization) * \
                             ( 1. - np.square(sin_theta * np.cos(self.phi_values)) ) + \
                             xaxis_polarization  * \
                             ( 1. - np.square(sin_theta * np.sin(self.phi_values)) )
            
        for pi in self.polar_intensities_iter:
            pi[:,:] *= correctn[:,:]

        return
    

    def intensity_profile(self):
        """
        Averages over the azimuth phi to obtain an intensity profile.

        Returns
        -------
        intensity_profile : ndarray, float
            An n x 2 array, where the first dimension is the magnitude |q| and
            the second is the average intensity at that point < I(|q|) >_phi.
        """

        intensity_profile      = np.zeros( (self.num_q, 2), dtype=np.float )
        intensity_profile[:,0] = self._q_values.copy()

        # average over shots, phi
        for pi_x in self.polar_intensities_iter:
            if self.polar_mask != None:
                pi_x *= self.polar_mask.astype(np.float)
            intensity_profile[:,1] += np.mean(pi_x, axis=1)
            
        intensity_profile[:,1] /= float(self.num_shots)

        return intensity_profile


    def correlate_intra(self, q1, q2, num_shots=0, normed=False, mean_only=True):
        """
        Does intRA-shot correlations for many shots.

        Parameters
        ----------
        q1 : float
            The |q| value of the first ring
        q2 : float
            The |q| value of the second ring

        Optional Parameters
        -------------------
        num_shots : int
            number of shots to compute correlators for
        normed : bool
            return the (std-)normalized correlation or un-normalized correlation
        mean_only : bool
            whether or not to return every correlation, or the average

        Returns
        -------
        intra : ndarray, float
            Either the average correlation, or every correlation as a 2d array
        """

        logger.info("Correlating rings at %f / %f" % (q1, q2))

        q_ind1 = self.q_index(q1)
        q_ind2 = self.q_index(q2)

        if num_shots == 0: # then do correlation for all shots
            num_shots = self.num_shots
            
        # generate an output space
        if mean_only:
            intra = np.zeros(self.num_phi)
        else:
            intra = np.zeros((num_shots, self.num_phi))

        # Check if mask exists
        if self.polar_mask != None:
            mask1 = self.polar_mask[q_ind1,:]
            mask2 = self.polar_mask[q_ind2,:]
        else:
            mask1 = None
            mask2 = None
        
        # if we request normalization, avg variances along the way
        if normed:
            var1 = 0.0
            var2 = 0.0
        
        for i,pi in enumerate(self.polar_intensities_iter):

            logger.info('Correlating shot %d/%d' % (i+1, num_shots))
            
            rings1 = pi[q_ind1,:]
            rings2 = pi[q_ind2,:]
            
            if mean_only:
                intra += self._correlate_rows(rings1, rings2, mask1, mask2)
            else:
                intra[i,:] = self._correlate_rows(rings1, rings2, mask1, mask2)
            
            if normed:
                var1 += np.var( rings1[mask1] )
                var2 += np.var( rings2[mask2] )
                
            if i == num_shots - 1:
                break
                
        if mean_only:
            intra /= float(num_shots)
            
        if normed:
            intra /= np.sqrt( var1 * var2 / np.square(float(num_shots)) )
            #assert intra.max() <=  1.1
            #assert intra.min() >= -1.1
        
        return intra
    

    def correlate_inter(self, q1, q2, num_pairs=0, normed=False, mean_only=True):
        """
        Does intER-shot correlations for many shots.

        Parameters
        ----------
        q1 : float
            The |q| value of the first ring
        q2 : float
            The |q| value of the second ring
        
        Optional Parameters
        -------------------
        num_pairs : int
            number of pairs of shots to compute correlators for
        normed : bool
            return the (std-)normalized correlation or un-normalized correlation
        mean_only : bool
            whether or not to return every correlation, or the average

        Returns
        -------
        inter : ndarray, float
            Either the average correlation, or every correlation as a 2d array
        """

        logger.info("Correlating rings at %f / %f" % (q1, q2))

        q_ind1 = self.q_index(q1)
        q_ind2 = self.q_index(q2)

        max_pairs = self.num_shots * (self.num_shots - 1) / 2
        
        if (num_pairs == 0) or (num_pairs > max_pairs):
            inter_pairs = utils.all_pairs(self.num_shots)
            num_pairs = max_pairs
        else:
            inter_pairs = utils.random_pairs(self.num_shots, num_pairs)
            
        # generate an output space
        if mean_only:
            inter = np.zeros(self.num_phi)
        else:
            inter = np.zeros((num_pairs, self.num_phi))

        # Check if mask exists
        if self.polar_mask != None:
            mask1 = self.polar_mask[q_ind1,:]
            mask2 = self.polar_mask[q_ind2,:]
        else:
            mask1 = None
            mask2 = None     

        # if we request normalization, avg variances along the way
        if normed:
            var1 = 0.0
            var2 = 0.0
        
        for k,(i,j) in enumerate(inter_pairs):
            
            logger.info('Correlating intra %d/%d' % (i+1, num_pairs))
            
            if self._polar_intensities_type == 'array':
                rings1 = self._polar_intensities[i,q_ind1,:]
                rings2 = self._polar_intensities[j,q_ind2,:]
                
            # todo : this could be very slow if we have to skip all over the
            #        place on disk -- do some tests, see if it's a problem,
            #        act accordingly
            elif self._polar_intensities_type == 'tables':
                rings1 = self._polar_intensities.read(i)
                rings2 = self._polar_intensities.read(j)
                rings1 = rings1[0,q_ind1,:]
                rings2 = rings2[0,q_ind2,:]
            
            if mean_only:
                inter += self._correlate_rows(rings1, rings2, mask1, mask2)
            else:
                inter[i,:] = self._correlate_rows(rings1, rings2, mask1, mask2)
            
            if normed:
                var1 += np.var( rings1[mask1] )
                var2 += np.var( rings2[mask2] )
            
                
        if mean_only:
            inter /= float(num_pairs)
            
        if normed:
            inter /= np.sqrt( var1 * var2 / np.square(float(num_pairs)) )
            #assert inter.max() <=  1.0
            #assert inter.min() >= -1.0

        return inter
        
        
    @staticmethod
    def _correlate_rows(x, y, x_mask=None, y_mask=None):
        """
        Compute the (unnormalized) circular correlation function across the rows
        of x,y. The correlation functions are computed using the fluctuations
        around each sample (shot-by-shot mean subtraction).
        
        Parameters
        ----------
        x,y : np.ndarray, float
            2D arrays of size N x M, where N indexes "experiments" and M indexes
            an observation vector for each experiment.

        Optional Parameters
        -------------------
        x_mask,y_mask : np.ndarray, bool
            Arrays describing masks over the data. These are 1D arrays of size
            M, with a single value for each data point.
        
        Returns
        -------
        corr : np.ndarray, float
            The N x M circular correlation function for each experiment. If
            `mean_only` is true, this is just a len-M array, averaged over
            the first dimension.
        """
        
        # do a shitload of typechecking -.-
        if len(x.shape) == 1:
            x = x[None,:]
            flatten = True
        elif len(x.shape) > 2:
            raise ValueError('`x` must be one or two dimensional array')
        else:
            flatten = False
            
        if len(y.shape) == 1:
            y = y[None,:]
        elif len(y.shape) > 2:
            raise ValueError('`y` must be one or two dimensional array')
            
        if not y.shape == x.shape:
            raise ValueError('`x`,`y` must have the same shape')

        n_row = x.shape[0]
        n_col = x.shape[1]

        if x_mask != None: 
            assert len(x_mask) == n_col
            x_mask = x_mask.astype(np.bool)

        if y_mask != None:
            assert len(y_mask) == n_col
            y_mask = y_mask.astype(np.bool)
            
        # actually compute some correlators
        if ((x_mask == None) and (y_mask == None)):
            xm = 1.0
            ym = 1.0
            N_delta = float(n_col) # normalization factor
            
        else:
            
            # if we only supply one mask...
            if x_mask == None:
                x_mask = np.ones_like(y_mask)
            elif y_mask == None:
                y_mask = np.ones_like(x_mask)
            
            xm = x_mask[None,:]
            ym = y_mask[None,:]
            
            # if we're masking, we have a delta-dependent normalization
            N_delta = np.zeros(n_col)
            for delta in range(n_col):
                N_delta[delta] = np.sum( xm * np.roll(ym, delta) )
            N_delta = N_delta[None,:]
            
        # use these for mean subtracting, not normalizing
        x_bar = x.mean(axis=1)[:,None]
        y_bar = y.mean(axis=1)[:,None]
        
        # use d-FFT + convolution thm
        ffx  = np.fft.rfft((x - x_bar) * xm, n=n_col, axis=1)
        ffy  = np.fft.rfft((y - y_bar) * ym, n=n_col, axis=1)
        corr = np.fft.irfft( ffx * np.conjugate(ffy), n=n_col, axis=1)
        assert corr.shape == (n_row, n_col)
                    
        # normalize by the number of pairs
        corr = np.select([corr != 0.0, corr == 0.0], [corr / N_delta, 0.0])
        
        if flatten:
            corr = corr.flatten()
        
        return corr
    

    def legendre(self, q1, q2, order, use_inter_statistics=False):
        """
        Project the correlation functions onto a set of legendre polynomials,
        and return the coefficients of that projection.

        Parameters
        ----------
        order : int
            The order at which to truncate the polynomial expansion. Note that
            this function projects only onto even numbered Legendre polynomials.

        Optional Parameters
        -------------------
        use_inter_statistics : bool
            Whether or not to subtract inter-shot statistics from the
            correlation function before projecting it. This can help remove
            detector artifacts at the cost of a small computational overhead.

        Returns
        -------
        c: np.ndarray, float
            An array of the legendre coefficients. Contains all coefficients
            (both even and odd) up to order `order`
        """

        # iterate over each pair of rings in the collection and project the
        # correlation between those two rings into the Legendre basis

        if use_inter_statistics:
            corr = self.correlate_intra(q1, q2, mean_only=True) - \
                   self.correlate_inter(q1, q2, mean_only=True, num_pairs=self.num_shots)
        else:
            corr = self.correlate_intra(q1, q2, mean_only=True)

        # tests indicate this is a good numerical projection
        c = np.polynomial.legendre.legfit(self.cospsi(q1, q2), corr, order-1)
        
        return c


    def legendre_matrix(self, order, use_inter_statistics=False):
        """
        Project the correlation functions onto a set of legendre polynomials,
        and return the coefficients of that projection.

        Parameters
        ----------
        order : int
            The order at which to truncate the polynomial expansion. Note that
            this function projects only onto even numbered Legendre polynomials.

        Optional Parameters
        -------------------
        use_inter_statistics : bool
            Whether or not to subtract inter-shot statistics from the
            correlation function before projecting it. This can help remove
            detector artifacts at the cost of a small computational overhead.

        Returns
        -------
        Cl: np.ndarray, float
            An array of the legendre coefficients. Contains all coefficients
            (both even and odd) up to order `order`. The returned object is a
            3-D array indexed by

                (order, q_ind1, q_ind2)

            where the q_ind values are the indices that map onto self.q_values.
        """

        # initialize space for coefficients
        Cl = np.zeros( (order, self.num_q, self.num_q) )

        # iterate over each pair of rings in the collection and project the
        # correlation between those two rings into the Legendre basis

        for i in range(self.num_q):
            q1 = self.q_values[i]
            for j in range(i,self.num_q):
                q2 = self.q_values[j]
                c  = self.legendre(q1, q2, order=order, 
                                   use_inter_statistics=use_inter_statistics)
                Cl[:,i,j] = c
                Cl[:,j,i] = c  # copy it to the lower triangle too

        return Cl


    @classmethod
    def simulate(cls, traj, num_molecules, q_values, num_phi, num_shots,
                 energy=10, traj_weights=None, force_no_gpu=False, 
                 photons_scattered_per_shot=None, device_id=0):
        """
        Simulate many scattering 'shot's, i.e. one exposure of x-rays to a
        sample, but onto a polar detector. Return that as a Rings object
        (factory function).

        Assumes we have a Boltzmann distribution of `num_molecules` identical
        molecules (`trajectory`), exposed to a beam defined by `beam` and
        projected onto `detector`.

        Each conformation is randomly rotated before the scattering simulation is
        performed. Atomic form factors from X, finite photon statistics, and the
        dilute-sample (no scattering interference from adjacent molecules)
        approximation are employed.

        Parameters
        ----------
        traj : mdtraj.trajectory
            A trajectory object that contains a set of structures, representing
            the Boltzmann ensemble of the sample. If len(traj) == 1, then we
            assume the sample consists of a single homogenous structure,
            replecated `num_molecules` times.

        num_molecules : int
            The number of molecules estimated to be in the `beam`'s focus.

        q_values : ndarray/list, float
            The values of |q| to extract rings at (in Ang^{-1}).

        num_phi : int
            The number of equally spaced points around the azimuth to
            interpolate onto (e.g. `num_phi`=360 means 1 deg spacing).

        num_shots : int
            The number of shots to perform and include in the Shotset.

        Optional Parameters
        -------------------
        energy : float
            The energy, in keV

        traj_weights : ndarray, float
            If `traj` contains many structures, an array that provides the
            Boltzmann weight of each structure. Default: if traj_weights == None
            weights each structure equally.

        force_no_gpu : bool
            Run the (slow) CPU version of this function.

        photons_scattered_per_shot : int
            The number of photons scattered to the detector per shot. For use
            with `finite_photon`. If "-1" (default), use continuous scattering
            (infinite photon limit).

        Returns
        -------
        rings : odin.xray.Rings
            A Rings instance, containing the simulated shots.
        """

        device_id = int(device_id)
        beam = Beam(photons_scattered_per_shot, energy=energy)
        k = beam.k
        q_values = np.array(q_values)

        qxyz = _q_grid_as_xyz(q_values, num_phi, k)

        # --- simulate the intensities ---

        polar_intensities = np.zeros((num_shots, len(q_values), num_phi))

        for i in range(num_shots):
            I = scatter.simulate_shot(traj, num_molecules, qxyz,
                                      traj_weights=traj_weights,
                                      finite_photon=photons_scattered_per_shot,
                                      force_no_gpu=force_no_gpu,
                                      device_id=device_id)
            polar_intensities[i,:,:] = I.reshape(len(q_values), num_phi)

            logger.info('Finished polar shot %d/%d on device %d' % (i+1, num_shots, device_id) )

        return cls(q_values, polar_intensities, k, polar_mask=None)


    def save(self, filename):
        """
        Saves the Rings object to disk.

        Parameters
        ----------
        filename : str
            The name of the file to write to disk. Must end in '.ring' -- if you
            don't put this, it will be automatically added.
        """

        if not filename.endswith('.ring'):
            filename += '.ring'
            
        if os.path.exists(filename):
            raise IOError('File: %s already exists! Aborting...' % filename)

        # if self.polar_mask == None, then save a single 0
        if self.polar_mask == None:
            pm = np.array([0])
        else:
            pm = self.polar_mask
            
        hdf = tables.File(filename, 'w')

        # these are going to be CArrays
        io.saveh( hdf,
                  q_values = self._q_values,
                  k = np.array([self.k]),
                  polar_mask = pm )
                  
        # but we want `polar_intensities` to be an EArray
        a = tables.Atom.from_dtype(np.dtype(np.float64))
        pi_node = hdf.createEArray(where='/', name='polar_intensities',
                                   shape=(0, self.num_q, self.num_phi), 
                                   atom=a, filters=io.COMPRESSION)
                                        
        for intx in self.polar_intensities_iter:
            pi_node.append(intx[None,:,:])
        
        hdf.close()

        logger.info('Wrote %s to disk.' % filename)

        return
    

    @classmethod
    def load(cls, filename, force_into_memory=False):
        """
        Load a Rings object from disk.

        Parameters
        ----------
        filename : str
            The name of the file to write to disk. Must end in '.ring'.
        
        Optional Parameters
        -------------------
        force_into_memory : bool
            Whether or not to load the entire dataset into memory (True) or
            use lazy loading (False, default).
                        
        Returns
        -------
        rings : xray.Rings
            The loaded rings object.
        """

        if not filename.endswith('.ring'):
            raise ValueError('Must load a rings file (.ring)')

        hdf = tables.File(filename, 'r+')
        
        q_values = hdf.root.q_values.read()
        pm = hdf.root.polar_mask.read()
        k = float(hdf.root.k.read()[0])

        # deal with our codified polar mask
        if np.all(pm == np.array([0])):
            pm = None
            
        if force_into_memory:
            pi_handle = hdf.root.polar_intensities.read()
            hdf.close()
            hdf = None
        else:
            pi_handle = hdf.root.polar_intensities
        
        rings = cls(q_values, pi_handle, k, polar_mask=pm)
        rings._hdf = hdf
        
        return rings
        
        
    def append(self, other_rings):
        """
        Combine a two rings objects. We keep the mask from the current Rings.
        
        Parameters
        ----------
        other_rings : xray.Rings
            A different rings object, but with the same q_values, num_phi.
            
        Returns
        -------
        None
        """
        
        if not np.all(other_rings.q_values == self.q_values):
            raise ValueError('Two rings must have exactly the same q_values')
        if not other_rings.k == self.k:
            raise ValueError('Two rings must have exactly the same wavenumber (k)')
            
        if self._polar_intensities_type == 'array':            
            combined_pi = np.vstack( (self.polar_intensities, other_rings.polar_intensities) )
            self._polar_intensities = combined_pi
            
        elif self._polar_intensities_type == 'tables':
            
            if not type(self._polar_intensities) == tables.earray.EArray:
                raise TypeError('Malformed Rings object : internal tables type'
                                ' for `polar_intensities` is non-extensible '
                                'CArray. EArrays are necessary to append data.'
                                ' This may be solved by first saving the Rings'
                                ' to disk, loading them back into memory with'
                                ' `force_into_memory` set to true, and then '
                                'saving/loading once more. The result *should*'
                                ' be an EArray version of the Rings object.')
            
            for pi_intx in other_rings.polar_intensities_iter:
                self._polar_intensities.append(pi_intx[None,:,:])
            
        return


def _q_grid_as_xyz(q_values, num_phi, k):
    """
    Generate a q-grid in cartesian space: (q_x, q_y, q_z).

    Parameters
    ----------
    q_values : ndarray/list, float
        The values of |q| to extract rings at (in Ang^{-1}).

    num_phi : int
        The number of equally spaced points around the azimuth to
        interpolate onto (e.g. `num_phi`=360 means 1 deg spacing).

    Returns
    -------
    qxyz : ndarray, float
        An N x 3 array of (q_x, q_y, q_z)
    """
    
    q_values = np.array(q_values)

    phi_values = np.linspace( 0.0, 2.0*np.pi, num=num_phi )
    num_q = len(q_values)

    # q_h is the magnitude projection of the scattering vector on (x,y)
    q_z = - np.power(q_values, 2) / (2.0 * k)
    q_h = q_values * np.sqrt( 1.0 - np.power( q_values / (2.0 * k), 2 ) )

    # construct the polar grid:
    qxyz = np.zeros(( num_q * num_phi, 3 ))
    qxyz[:,0] = np.repeat(q_h, num_phi) * np.cos(np.tile(phi_values, num_q)) # q_x
    qxyz[:,1] = np.repeat(q_h, num_phi) * np.sin(np.tile(phi_values, num_q)) # q_y
    qxyz[:,2] = np.repeat(q_z, num_phi)                                      # q_z

    return qxyz


def load(filename):
    """
    Load a file from disk, into a format corresponding to an object in 
    odin.xray. Includes readers for {.dtc, .shot, .ring}.
    
    Parameters
    ----------
    filename : str
        The path to a file.
    
    Returns
    -------
    obj : generic
        An odin object, whos type depends on the file extension.
    """
    
    if filename.endswith('.dtc'):
        obj = Detector.load(filename)
    elif filename.endswith('.shot'):
        obj = Shotset.load(filename)
    elif filename.endswith('.ring'):
        obj = Rings.load(filename)
    else:
        raise IOError('Could not understand format of file: %s. Extension must '
                      ' be one of {.dtc, .shot, .ring}.' % filename)
        
    return obj

