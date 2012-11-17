
from mdtraj import trajectory
from odin import structure
from odin.testing import ref_file

import numpy as np
from numpy.testing import assert_array_almost_equal

import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logging.basicConfig()

def test_m_confs():
    t = trajectory.load( ref_file('ala2.pdb') )
    m = structure.multiply_conformations(t, 10, 1.0)
    return m
    
def test_rm_com():
    t = trajectory.load( ref_file('ala2.pdb') )
    r = structure.remove_COM(t)
    
    masses = [ a.element.mass for a in t.topology.atoms() ]
    
    for i in range(t.n_frames):
        assert_array_almost_equal(np.zeros(3), 
                                  np.average(t.xyz[i,:,:], weights=masses, axis=0))
    
    
if __name__ == '__main__':
    m = test_m_confs()
    print m.n_frames
    m.save('test.pdb')