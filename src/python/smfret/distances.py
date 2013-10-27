"""
this file contains utility functions for calculating the distance
between donor and acceptor in a FRET experiment. There are many 
ways to do this, with varying levels of rigor.
"""
from mdtraj import geometry
import numpy as np
import itertools

def atom_atom(trajectory, inds):
    """
    Approximate the donor-acceptor distance as the distance between
    two particular atoms.

    Parameters
    ----------
    trajectory : mdtraj.Trajectory
        trajectory object with conformations
    inds : array_like
        length two array or list with the atom indices to calculate
        the donor-acceptor distances.

    Returns
    -------
    distances : np.ndarray
        numpy array corresponding to the length between two atoms
        for each conformation in `trajectory`
    """
    
    inds = np.array(inds)
    if inds.shape != (2,):
        raise Exception("inds must contain only two indices")

    return geometry.distance.compute_distances(trajectory, atom_pairs=np.array([inds]),
        periodic=False)


def residue_residue(trajectory, inds, scheme='min'):
    """
    Approximate the donor-acceptor distance as the distance between
    two residues' heavy atoms. 

    Parameters
    ----------
    trajectory : mdtraj.Trajectory
        trajectory object with conformations
    inds : array_like
        length two array or list with the atom indices to calculate
        the donor-acceptor distances.
    scheme : {'mean', 'min', 'max'}
        calculate the distance between all heavy atoms in the two
        residues and use this scheme to provide an estimate of the
        donor-acceptor distance (Default: 'min')

    Returns
    -------
    distances : np.ndarray
        numpy array corresponding to the length between two atoms
        for each conformation in `trajectory`
    """

    inds = np.array(inds)
    if inds.shape != (2,):
        raise Exception("inds must contain only two indices")

    scheme = scheme.lower()
    if not scheme in ['mean', 'min', 'max']:
        raise Exception("scheme must be one of ('mean', 'min', 'max')")

    if scheme == 'mean':
        scheme = lambda ary : ary.mean(1)
    elif scheme == 'min':
        scheme = lambda ary : ary.min(1)
    elif scheme == 'max':
        scheme = lambda ary : ary.max(1)

    res0 = trajectory.topology.residue(inds[0])
    res1 = trajectory.topology.residue(inds[1])

    heavy_atom_inds0 = [a.index for a in res0.atoms if a.element.symbol != 'H']
    heavy_atom_inds1 = [a.index for a in res1.atoms if a.element.symbol != 'H']

    atom_pairs = np.array(list(itertools.product(heavy_atom_inds0, heavy_atom_inds1)))

    atom_distances = geometry.distance.compute_distances(trajectory, 
        atom_pairs=atom_pairs, periodic=False)

    distances = scheme(atom_distances)

    return distances
    

