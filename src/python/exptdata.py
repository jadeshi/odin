
"""
exptdata.py

Experimental Data
"""

import os
from glob import glob
import numpy as np

from mdtraj import io


class ExptDataCollection(object):
    """
    A collection of ExptData classes, of various types. This class provides an
    aggregated interface to all those data.
    """
    
    def __init__(self, expt_data_list):
        """
        Generate an instance of the class.
        
        Parameters
        ----------
        expt_data_list : list
            A list of various ExptData instances, mapping to all the data to
            include in this structure prediction round.
        """
        
        self._n_data = np.sum([ d._n_data for d in expt_data_list ])
        self._directories = [d._directory for d in expt_data_list]
        
        self._files = {}
        self._expttype = []
        self._exptmeta = []
        values         = []
        errors         = []
        for d in expt_data_list:
            self._files[d._directory] = d._files
            
            if isinstance(d, ScatteringData):
                self._expttype.append('scattering')
            elif isinstance(d, ChemShiftData):
                self._expttype.append('chemshift')
            else:
                raise RuntimeError('Invalid experimental data type. Currently'
                                   'allowed: ScatteringData, ChemShiftData.')
                                   
            self._exptmeta.append(d._exptmeta)
            
            values.append(d._values)
            errors.append(d._errors)
            
        self._values = np.array(values)
        self._errors = np.array(errors)
        
        assert len(self._values) == self._n_data
        assert len(self._errors) == self._n_data
        assert len(self._exptmeta) == self._n_data

    @property
    def n_data(self):
        return self._n_data

    @property
    def directories(self):
        return self._directories
        
    @property
    def files(self):
        return self._files
        
    @property
    def exptmeta(self):
        """
        Metadata necessary for identifying what an experimental value is. Should
        include all information necessary to provide a prediction of that
        experiment, when the x,y,z coordinates of the system are also provided.
        
        E.g., for chemical shifts this would be the atom (index, element type),
        for scattering it would be the detector pixel location (q-vector), etc.
        """
        return self._exptmeta
        
    @property
    def expttype(self):
        """
        For each data point, the kind of experiment that generated that data
        """
        return self._expttype
        
    @property
    def values(self):
        """
        The measured values of each data point
        """
        return self._values
        
    @property
    def errors(self):
        """
        The errors associated with each experiment.
        """
        return self._errors
        

    def save(self, target):
        """
        Save all experimental data to disk, in one file.
        
        This file needs to contain:
        
        MeasurementID | ExptType | Metadata | Value | Error
        
        """
        
        if os.path.exists(target):
            raise ValueError('File exists! %s' % target)
        
        if target.endswith('.db'):
            logger.info('Writing expt. data to database: %s' % target)
            self._to_sqlite(target)
        elif target.endswith('.hdf'):
            logger.info('Writing expt. data to HDF5: %s' % target)
            self._to_file(target)
        else:
            raise ValueError('Cannot understand what format to write %s to. Options: .hdf, .db.' % target)

    def load(self, target):
        pass
        
    def _to_sqlite(self, dbfilename):
        pass
        
    def _from_sqlite(self, dbfilename):
        pass
        
    def _to_file(self, filemame):
        # use Cpickle, binary serialization
    
    @classmethod
    def _from_file(self, filename):
        pass


class ExptData(object):
    """
    Abstract base class for experimental data classes.
        
    All ExptData objects should have the following properties:
    -- values (the experimental values)
    -- errors (the STD error associated with values)
        
    Further, each ExptData inheretant should provide a self.predict(xyz) method,
    that outputs an array of len(values) that is the *prediction* of the array
    `values` given a molecular geometry `xyz`.
    """
        
    def __init__(self, directory):
        """
        Initialize the class, loading into memory all experimental data.
        """
        
        self._directory = directory
        
        self._files = []
        for filetype in self.acceptable_filetypes:
            self._files += glob(directory + '/*' + filetype)
            
        for fn in self._files:
            self._load_file(fn)
        
        self._n_data = len(self._values)
        
        assert(len(self._errors) == self._n_data)
        assert(len(self._metadata) == self._n_data)
        
    @property
    def n_data(self):
        return self._n_data
        
    @property
    def directory(self):
        return self._directory
        
    @property
    def files(self):
        return self._files
        
    @property
    def values(self):
        """
        The measured values of each data point
        """
        return self._get_values()
        
    @property
    def errors(self):
        """
        The errors associated with each experiment.
        """
        return self._get_errors()
        
    @property
    def acceptable_filetypes(self):
        """
        A list object of the file extensions this class knows how to load.
        """
        return self._acceptable_filetypes()

    # TJL : need to figure out how ABC's work. 
    # Classes that inherent from ExptData should implement all the methods below 
    @abstractplaceholder
    def _load_file(self, filename):
        """
        Load a file containing experimental data, and dump the relevant data,
        errors, and metadata into the object in a smart/comprehendable manner.
        """
        return
        
    @abstractplaceholder
    def predict(self, trajectory):
        """
        Method to predict the array `values` for each snapshot in `trajectory`.
        
        Parameters
        ----------
        trajectory : mdtraj.trajectory
            A trajectory to predict the experimental values for.
        
        Returns
        -------
        prediction : ndarray, 2-D
           The predicted values. Will be two dimensional, 
           len(trajectory) X len(values).
        """
        return prediction
        
    @abstractplaceholder
    def _default_error(self):
        """
        Method to estimate the error of the experiment (conservatively) in the
        absence of explicit input.
        """
        return error_guess
        
    @abstractplaceholder
    def _get_values(self):
        """
        Return an array `values`, in an order that ensures it will match up
        with the method self.predict()
        """
        return values

    @abstractplaceholder
    def _get_errors(self):
        """
        Return an array `errors`, in an order that ensures it will match up
        with the method self.predict()
        """
        return errors
    
    @abstractplaceholder    
    def _acceptable_filetypes(self):
        """
        A list object of the file extensions this class knows how to load.
        """
        return list_of_extensions
        

class ScatteringData(ExptData):
    """
    A class supporting x-ray scattering data.
    
    Two kinds of experimental observables are currently supported by this class
    and it's predict() method:
    
    -- intensity profiles
         The observed x-ray scattering averaged over the azimuthal (radial)
         angle on the detector. Also known as SAXS/WAXS data.
    -- correlations
         This is the correlation function of an intensity 'ring' on the detector.
         These correlations promise to hold more structural information than
         the intensity profile alone, but have an intrinsically greater error.
         
    The intensity profile is stored in self._intensity profile. This is the
    average profile over all data files that get loaded. self._intensity is
    a 2-d array, the first dimension is the list of q-values at which the
    intensities were measured, the second is the intensity value at that q.
    
    The correlation data is stored in a 3-d array. Each element of that array
    contains one value of the correlation function C(q_1, q_2, phi). Each
    dimension scans across each parameter of that function C
    
        Dim 0: q_1 magnitude
        Dim 1: q_1 magnitude
        Dim 2: phi
    
    The values that each index corresponds to are held in two other arrays:
        
        self._correlated_q_mags
        self._correlated_phis
    
    Such that, for example,
    
        self._correlations[2,3,10]
    
    corresponds to C(q1, q2, phi) for
    
        q1  = self._correlated_q_mags[2]
        q1  = self._correlated_q_mags[3]
        phi = self._correlated_phis[10]
        
    This mirrors the structure of the return value of 
    xray.Shotset.correlate_all_rings(), which ultimately generates the data
    arrays for this class.
    """
        
    def _load_file(self, filename):
        """
        Load a file containing experimental data, and dump the relevant data,
        errors, and metadata into the object in a smart/comprehendable manner.
        """
        return
        
        
    def predict(self, trajectory):
        """
        Method to predict the array `values` for each snapshot in `trajectory`.
        
        Parameters
        ----------
        trajectory : mdtraj.trajectory
            A trajectory to predict the experimental values for.
        
        Returns
        -------
        prediction : ndarray, 2-D
           The predicted values. Will be two dimensional, 
           len(trajectory) X len(values).
        """
        return prediction
        
        
    def _default_error(self):
        """
        Method to estimate the error of the experiment (conservatively) in the
        absence of explicit input.
        """
        return error_guess
        
        
    def _get_values(self):
        """
        Return an array `values`, in an order that ensures it will match up
        with the method self.predict()
        """
        return values
        
        
    def _get_errors(self):
        """
        Return an array `errors`, in an order that ensures it will match up
        with the method self.predict()
        """
        return errors
        
     
    def _acceptable_filetypes(self):
        """
        A list object of the file extensions this class knows how to load.
        """
        list_of_extensions = ['cxi', 'cbf', 'shot']
        return list_of_extensions
        
        
    def _predict_intensity_profile(self, trajectory):
        """
        Return the predicted intensity profile, using the Debye equation.
        Returns just the intensities, not the q-values (a 1-d array)
        """
        ip = xray.debye(trajectory, q_values=self._intensity_profile[:,0])
        return ip[:,1]
        
        
    def _predict_correlations(self, trajectory):
        """
        """
        pass

        
        

class ChemShiftData(ExptData):
    """
    A class supporting chemical shift data
    """
    pass