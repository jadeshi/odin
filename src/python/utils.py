
"""
Functions that are useful in various places, but have no common theme.
"""

import functools

from pprint import pprint
from argparse import ArgumentParser
from simtk import unit

import numpy as np


class odinparser(ArgumentParser):
    """
    Simple extension of argparse, designed to automatically print stuff
    """
    def parse_args(self):
        print graphic
        args = super(odinparser, self).parse_args()
        pprint(args.__dict__)
        return args
    
        
def is_iterable(obj):
    """
    Determine if `obj` is iterable. Returns bool.
    """
    try:
        for x in obj:
            pass
        return True
    except:
        return False
    
        
# http://stackoverflow.com/questions/11984684/display-only-one-logging-line
logger_return = '\x1b[80D\x1b[1A\x1b[K'

    
def all_pairs(n):
    """
    Generator that yields all unique pairs formed by the integers {0, ..., n-1}
    """
    for i in xrange(n):
        for j in xrange(i+1,n):
            yield (i,j)
    
    
def unique_rows(a):
    """
    For a two-dim array, returns unique rows.
    """
    unique_a = np.unique(a.view([('', a.dtype)]*a.shape[1]))
    return unique_a.view(a.dtype).reshape((unique_a.shape[0], a.shape[1]))
    

def random_pairs(total_elements, num_pairs): #, extra=10):
    """
    Sample `num_pairs` random pairs (i,j) from i,j in [0:total_elements) without
    replacement.
    
    Parameters
    ----------
    total_elements : int
        The total number of elements.
    num_pairs : int
        The number of unique pairs to sample
    
    Returns
    -------
    pairs : np.ndarray, int
        An `num_pairs` x 2 array of the random pairs.
    
    if num_pairs > (total_elements * (total_elements-1)) / 2:
        raise ValueError('Cannot request more than N(N-1)/2 unique pairs')
    
    not_done = True
    
    while not_done:
        n_to_draw = num_pairs + extra
        p = np.random.randint(0, total_elements, size=(num_pairs, 2))
        p.sort(axis=1)
        p = unique_rows(p)
        p = p[ p[:,0] != p[:,1] ] # slice out i == j
        
        if p.shape[0] >= num_pairs:
            p[:num_pairs]
            not_done = False
        else:
            extra += 10
    
    return p[0:num_pairs]
    """
    
    np.random.seed()
    inter_pairs = []
    factor = 2
    while len(inter_pairs) < num_pairs:
        rand_pairs   = np.random.randint( 0, total_elements, (num_pairs*factor,2) )
        unique_pairs = list( set( tuple(pair) for pair in rand_pairs ) )
        inter_pairs  = filter( lambda x:x[0] != x[1], unique_pairs)
        factor += 1
        
    return np.array(inter_pairs[0:num_pairs])


def maxima(a):
    """
    Returns the indices where `a` is at a local max.
    """
    return np.where(np.r_[True, a[1:] > a[:-1]] & np.r_[a[:-1] > a[1:], True] == True)[0]
        

def get_simulation_xyz(simulation):
    """
    get the current state of an OpenMM simulation and return
    the system coordinates as a numpy array
    """

    pos = simulation.context.getState(getPositions=True).getPositions()

    n_atoms = len(pos)
    xyz = np.zeros((n_atoms, 3))

    for i in range(n_atoms):
        for j in range(3):
            xyz[i,j] = pos[i][j].value_in_unit(unit.nanometers)

    return xyz

    
def write_sample_input(filename='sample.odin'):
    txt=''' # THIS FILE WAS GENERATED BY ODIN -- sample input file
    
    runname: testrun                 # used to name directories, etc.
    
    
    # RUN SETTINGS 
    predict:   boltzmann             # {single, boltzmann, kinetic} ensembles
    sampling:  md                    # either {md, mc} for dynamics, Monte Carlo
    prior:     minimal               # could be amber99sb-ildn, charm22, etc.
    solvent:   none                  # grand cannonical solvent to employ
    outputdir: ~/testrun             # where stuff gets written -- could be GB.
    
    
    # EXPERIMENTS
    experiment: LCLS_run_1           # a name identifying the data
        - dir:  ~/testrun/lcls       # should contain pre-processed data files
        - type: scattering           # {scattering, chemshift} are in now
    
    experiment: NMR_HSQC_1
        - dir:  ~/testrun/chemshifts
        - type: chemshift
        
        
    # RESOURCES
    runmode: cluster                 # one of {local, cluster}
    nodes:   4                       # how many nodes to call for
    gpn:     1                       # gpus per node
    REMD:    True                    # use REMD to estimate the lambdas
    temps:   [1, 0.5, 0.1, 0.01]     # temps in units of beta, <= nodes*gpn
    
    '''
    f = open(filename, 'w')
    f.write(txt)
    f.close()
    logger.info("Wrote: %s" % filename)
    return

    
def memoize(obj):
    """
    An expensive memoizer that works with unhashables
    """
    # stolen unashamedly from pymc3
    cache = obj.cache = {}
    
    @functools.wraps(obj)
    def memoizer(*args, **kwargs):
        key = (hashable(args), hashable(kwargs))
        
        if key not in cache:
            cache[key] = obj(*args, **kwargs)
            
        return cache[key]
    
    return memoizer

    
def hashable(a):
    """
    Turn some unhashable objects into hashable ones.
    """
    # stolen unashamedly from pymc3
    if isinstance(a, dict):
        return hashable(a.iteritems())
    try:
        return tuple(map(hashable,a))
    except:
        return a

        
graphic = """
	                                    .  
	                                  .#   
	           .                     .##   
	          #.                     #  :  
	       .# .                     .# .#  
	       .. .       ..##.         #   #  
	       #  .     .#.    #       #    #  
	             .#         #.    #    #   
	      #    #             #.  #.    #   
	      # .#                ##      #    
	      ##.                #.      :#       ____  _____ _____ _   _ 
	      #                 .# .:   .#       / __ \|  __ \_   _| \ | |
	   .:.      .    .      .#..   .#       | |  | | |  | || | |  \| |
	  .####  . . ...####     #.   #.        | |  | | |  | || | | . ` |
	 # .##   . # . #.#   . =# .##.#         | |__| | |__| || |_| |\  |
	 . .##   .  #   ..   # = .=#  #          \____/|_____/_____|_| \_|
	#   . ####     .###,  ,      ##        
	#.## .               '#.. #    #                       Observation
	 .                      ##.. . #       	               Driven
	                          .#   #                       Inference
	                            #. /                   of eNsembles
		                        .#        

     ----------------------------------------------------------------------
"""
