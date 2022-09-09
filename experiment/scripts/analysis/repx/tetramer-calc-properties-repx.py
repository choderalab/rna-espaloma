#!/usr/bin/env python
# coding: utf-8
import os, sys, math
import numpy as np
import glob
import mdtraj
import logging
import netCDF4 as nc
import warnings
import pandas as pd
import logging
import openmmtools as mmtools
from pymbar import timeseries
from openmm import *
from openmm.app import *
import barnaba as bb
from barnaba import definitions
from barnaba.nucleic import Nucleic


logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore")


# ==============================================================================
# Extract trajectory from NetCDF4 file
# https://github.com/choderalab/yank/blob/master/Yank/analyze.py
# ==============================================================================
def extract_trajectory(reference, nc_path, nc_checkpoint_file=None, checkpoint_interval=50, state_index=None, replica_index=None,
                       start_frame=0, end_frame=-1, skip_frame=1, keep_solvent=True,
                       discard_equilibration=False):
    """Extract phase trajectory from the NetCDF4 file.
    Parameters
    ----------
    reference : str
        Path to reference pdb file
    nc_path : str
        Path to the primary nc_file storing the analysis options
    nc_checkpoint_file : str or None, Optional
        File name of the checkpoint file housing the main trajectory
        Used if the checkpoint file is differently named from the default one chosen by the nc_path file.
        Default: None
    checkpoint_interval : int >= 1, Default: 50
        The frequency at which checkpointing information is written relative to analysis information.
        This is a multiple
        of the iteration at which energies is written, hence why it must be greater than or equal to 1.
        Checkpoint information cannot be written on iterations which where ``iteration % checkpoint_interval != 0``.
    state_index : int, optional
        The index of the alchemical state for which to extract the trajectory.
        One and only one between state_index and replica_index must be not None
        (default is None).
    replica_index : int, optional
        The index of the replica for which to extract the trajectory. One and
        only one between state_index and replica_index must be not None (default
        is None).
    start_frame : int, optional
        Index of the first frame to include in the trajectory (default is 0).
    end_frame : int, optional
        Index of the last frame to include in the trajectory. If negative, will
        count from the end (default is -1).
    skip_frame : int, optional
        Extract one frame every skip_frame (default is 1).
    keep_solvent : bool, optional
        If False, solvent molecules are ignored (default is True).
    discard_equilibration : bool, optional
        If True, initial equilibration frames are discarded (see the method
        pymbar.timeseries.detectEquilibration() for details, default is False).
    Returns
    -------
    trajectory: mdtraj.Trajectory
        The trajectory extracted from the netcdf file.
    """
    # Check correct input
    if (state_index is None) == (replica_index is None):
        raise ValueError('One and only one between "state_index" and '
                         '"replica_index" must be specified.')
    if not os.path.isfile(nc_path):
        raise ValueError('Cannot find file {}'.format(nc_path))

    # Import simulation data
    reporter = None
    try:
        reporter = mmtools.multistate.MultiStateReporter(nc_path, open_mode='r', checkpoint_storage=nc_checkpoint_file, checkpoint_interval=checkpoint_interval)
        reference = mdtraj.load_pdb(reference)
        topology = reference.topology
        
        # Get dimensions
        # Assume full iteration until proven otherwise
        last_checkpoint = True
        trajectory_storage = reporter._storage_checkpoint
        if not keep_solvent:
            # If tracked solute particles, use any last iteration, set with this logic test
            full_iteration = len(reporter.analysis_particle_indices) == 0
            if not full_iteration:
                trajectory_storage = reporter._storage_analysis
                topology = topology.subset(reporter.analysis_particle_indices)

        n_iterations = reporter.read_last_iteration(last_checkpoint=last_checkpoint)
        n_frames = trajectory_storage.variables['positions'].shape[0]
        n_atoms = trajectory_storage.variables['positions'].shape[2]
        logger.info('Number of frames: {}, atoms: {}'.format(n_frames, n_atoms))

        # Determine frames to extract.
        # Convert negative indices to last indices.
        if start_frame < 0:
            start_frame = n_frames + start_frame
        if end_frame < 0:
            end_frame = n_frames + end_frame + 1
        frame_indices = range(start_frame, end_frame, skip_frame)
        if len(frame_indices) == 0:
            raise ValueError('No frames selected')
        logger.info('Extracting frames from {} to {} every {}'.format(start_frame, end_frame, skip_frame))

        # Discard equilibration samples
        if discard_equilibration:
            u_n = extract_u_n(reporter._storage_analysis)
            # Discard frame 0 with minimized energy which throws off automatic equilibration detection.
            n_equil_iterations, g, n_eff = timeseries.detectEquilibration(u_n[1:])
            n_equil_iterations += 1
            logger.info(("Discarding initial {} equilibration samples (leaving {} "
                         "effectively uncorrelated samples)...").format(n_equil_iterations, n_eff))
            # Find first frame post-equilibration.
            if not full_iteration:
                for iteration in range(n_equil_iterations, n_iterations):
                    n_equil_frames = reporter._calculate_checkpoint_iteration(iteration)
                    if n_equil_frames is not None:
                        break
            else:
                n_equil_frames = n_equil_iterations
            frame_indices = frame_indices[n_equil_frames:-1]
        else:
            logging.info("Discard automatic equilibration detection")

        # Determine the number of frames that the trajectory will have.
        if state_index is None:
            n_trajectory_frames = len(frame_indices)
        else:
            # With SAMS, an iteration can have 0 or more replicas in a given state.
            # Deconvolute state indices.
            state_indices = [None for _ in frame_indices]
            for i, iteration in enumerate(frame_indices):
                replica_indices = reporter._storage_analysis.variables['states'][iteration, :]
                state_indices[i] = np.where(replica_indices == state_index)[0]
                #print(state_index, replica_indices, np.where(replica_indices == state_index)[0])
            n_trajectory_frames = sum(len(x) for x in state_indices)

        # Initialize positions and box vectors arrays.
        # MDTraj Cython code expects float32 positions.
        positions = np.zeros((n_trajectory_frames, n_atoms, 3), dtype=np.float32)
        box_vectors = np.zeros((n_trajectory_frames, 3, 3), dtype=np.float32)

        # Extract state positions and box vectors.
        if state_index is not None:
            logger.info('Extracting positions of state {}...'.format(state_index))

            # Extract state positions and box vectors.
            frame_idx = 0
            for i, iteration in enumerate(frame_indices):
                for replica_index in state_indices[i]:
                    positions[frame_idx, :, :] = trajectory_storage.variables['positions'][iteration, replica_index, :, :].astype(np.float32)
                    box_vectors[frame_idx, :, :] = trajectory_storage.variables['box_vectors'][iteration, replica_index, :, :].astype(np.float32)
                    frame_idx += 1

        else:  # Extract replica positions and box vectors
            logger.info('Extracting positions of replica {}...'.format(replica_index))

            for i, iteration in enumerate(frame_indices):
                positions[i, :, :] = trajectory_storage.variables['positions'][iteration, replica_index, :, :].astype(np.float32)
                box_vectors[i, :, :] = trajectory_storage.variables['box_vectors'][iteration, replica_index, :, :].astype(np.float32)
    finally:
        if reporter is not None:
            reporter.close()

    # Create trajectory object
    logger.info('Creating trajectory object...')
    trajectory = mdtraj.Trajectory(positions, topology)
    trajectory.unitcell_vectors = box_vectors

    return trajectory


if __name__ == '__main__':
    UNIT_NM_TO_ANGSTROMS = 10
    UNIT_PS_TO_NS = 1/1000

    # parameters to read repx trajectories
    CHECKPOINT_INTERVAL = 50
    START_FRAME = 1
    END_FRAME = -1
    SKIP_FRAME = 1
    DISCARD_EQUILIBRATION=False


    basepath = "../../../../"
    init_pdb = os.path.join(basepath, "eq/solvated.pdb")   # initial structure
    ref_pdb = os.path.join(basepath, "eq/min.pdb")         # reference structure
    ref_traj = mdtraj.load(ref_pdb)
    eq_ncfile = os.path.join(basepath, "eq/traj.nc")       # equilibrated
    eq_traj = mdtraj.load(eq_ncfile, top=init_pdb)
    ncfiles = glob.glob("../*/enhanced.nc")                # enhanced
    ncfiles.sort()
    print(ncfiles)


    # check number of states and replicas
    reporter = mmtools.multistate.MultiStateReporter(ncfiles[0], open_mode='r')
    n_states = reporter.n_states
    n_replicas = reporter.n_replicas
    print("n_states: {}".format(n_states))
    print("n_replicas: {}".format(n_replicas))


    # residue names
    rnames = [ residue.name for residue in ref_traj.topology.residues if residue.name not in ["HOH", "NA", "CL"]]
    print(rnames)


    # Load trajectories and calculate properties
    bb_angles = []
    pucker_angles = []
    rg = []
    rmsd, ermsd = [], []
    stackings = []
    dist1, dist2, dist3 = [], [], []
    couplings = []

    for ncfile in ncfiles:
        for i in range(n_states):    
            #print(i)
            t = extract_trajectory(reference=init_pdb, nc_path=ncfile, start_frame=START_FRAME, end_frame=END_FRAME, skip_frame=SKIP_FRAME, state_index=i, discard_equilibration=DISCARD_EQUILIBRATION)
            
            # backbone angles
            _angles, _res = bb.backbone_angles_traj(t)
            bb_angles.append(_angles)
            
            # suger pucker angles
            _angles, _res = bb.pucker_rao_traj(t)
            pucker_angles.append(_angles)
            
            # radius of gyration
            atom_indices = t.topology.select('not (protein or water or symbol Na or symbol Cl)')
            _t = t.atom_slice(atom_indices)
            rg.append(mdtraj.compute_rg(_t))
            
            # rmsd
            _rmsd = list(bb.functions.rmsd_traj(ref_traj, t))
            rmsd.append(_rmsd)
            
            # ermsd
            _ermsd = list(bb.functions.ermsd_traj(ref_traj, t))   
            ermsd.append(_ermsd)
            
            # annotation
            _stackings, _pairings, _res = bb.annotate_traj(t)
            stackings.append(_stackings)
            
            # N1 distance
            atom_indices = t.topology.select('name N1')
            # distance between first (5'-end) and second N1 atom
            _dist1 = mdtraj.compute_distances(t, atom_pairs=atom_indices[:2].reshape(1,2))
            dist1.append(_dist1)
            # distance between second and third (3'-end) N1 atom
            _dist2 = mdtraj.compute_distances(t, atom_pairs=atom_indices[1:3].reshape(1,2))
            dist2.append(_dist2)
            # distance between third and fourth (3'-end) N1 atom
            _dist3 = mdtraj.compute_distances(t, atom_pairs=atom_indices[2:].reshape(1,2))
            dist3.append(_dist3)
            
            # jcoupling
            _couplings, _res = bb.jcouplings_traj(t, couplings=["H1H2", "H2H3", "H3H4", "1H5P", "2H5P", "1H5H4", "2H5H4", "H3P"] )
            couplings.append(_couplings)
    
    # concatenate or flatten
    bb_angles = np.concatenate(bb_angles)
    pucker_angles = np.concatenate(pucker_angles)
    rg = np.concatenate(rg) * UNIT_NM_TO_ANGSTROMS
    rmsd = np.array(rmsd).flatten() * UNIT_NM_TO_ANGSTROMS
    ermsd = np.array(ermsd).flatten() * UNIT_NM_TO_ANGSTROMS
    stackings = np.concatenate(stackings)
    dist1 = np.concatenate(dist1) * UNIT_NM_TO_ANGSTROMS
    dist2 = np.concatenate(dist2) * UNIT_NM_TO_ANGSTROMS
    dist3 = np.concatenate(dist3) * UNIT_NM_TO_ANGSTROMS
    couplings = np.concatenate(couplings)

    assert bb_angles.shape[0] == pucker_angles.shape[0] == rg.shape[0] == rmsd.shape[0] == ermsd.shape[0] == stackings.shape[0] == dist1.shape[0] == dist2.shape[0] == dist3.shape[0], \
    print(bb_angles.shape, pucker_angles.shape, rg.shape, rmsd.shape, ermsd.shape, stackings.shape, dist1.shape, dist2.shape, dist3.shape)

    np.savez("mydata.npz", bb_angles=bb_angles, pucker_angles=pucker_angles, rg=rg, rmsd=rmsd, ermsd=ermsd, stackings=stackings, n1_dist=[dist1, dist2, dist3], couplings=couplings)

    #npzfile = np.load(outfile)
    #npzfile["bb_angles"]
