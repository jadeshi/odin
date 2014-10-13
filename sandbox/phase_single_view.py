#!/usr/bin/env python

"""
Run this on a machine with a fast nvidia GPU and OpenMM installed.
"""

import mdtraj
import thor
from thor import scatter

from odin import cdi
from odin import sample

# ------------------------------------------
# manually set params
#
test_system = 'ala' # 'ala' OR 'lyz'
prior       = 'amber99min.xml'
num_moves   = 1000
sigma       = 1000.0 # hmmmm
# ------------------------------------------

if test_system == 'ala':
    structure_file = 'ala2.pdb'
    dtc = thor.Detector.generic()
    
elif test_system == 'lyz':
    structure_file = '3lyz.clean.pdb'
    dtc = thor.Detector.load('lcls_test.dtc')
    
else:
    raise KeyError('no known test system: %s' % test_system)
    
    
starting_structure = mdtraj.load(structure_file)
qxyz = dtc.reciprocal
intensities = scatter.simulate_shot(starting_structure, 1, qxyz)

potential = cdi.CdiPotential(intensities, qxyz, sigma)

sampler = sample.MDMC(potential, prior, 
                      starting_structure.top, 
                      starting_structure.xyz[0])
               
sampler.sample(num_moves, 'test_traj.h5')

t = mdtraj.load('test_traj.h5')
t.save_pdb('test_traj.pdb')

