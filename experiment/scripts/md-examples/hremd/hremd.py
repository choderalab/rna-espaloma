#!/usr/bin/env python
# coding: utf-8


import os, sys, math
import numpy as np
import click
import inspect
from sys import stdout
import glob
import tempfile
from openmm import *
from openmm.app import *
from openmm.unit import *
from openmmtools import testsystems, states, mcmc, forces, alchemy
from openmmtools.utils import get_fastest_platform
from openmmtools.multistate import ReplicaExchangeSampler, MultiStateReporter
import mdtraj
import logging
import datetime
import warnings
import barnaba as bb
from barnaba import definitions
from barnaba.nucleic import Nucleic



def run(**options):
    #------------------------------------
    #system_options = {}
    #system_options['nonbondedMethod'] = PME
    #system_options['ewaldErrorTolerance'] = 0.0005    # default: 0.0005
    #system_options['nonbondedCutoff'] = 10 * angstroms  # default: 10 angstroms
    #system_options['rigidWater'] = True                # default: 
    #system_options['constraints'] = HBonds
    #system_options['hydrogenMass'] = 3.5 * amu

    default_pressure = 1 * atmosphere
    default_temperature = 275 * kelvin
    default_timestep = 4 * femtosecond
    default_collision_rate = 1/picosecond
    default_swap_scheme = 'swap-all'
    default_steps_per_replica = 250   # 1 ps/replica
    default_number_of_iterations = 30000  # 30 ns/replica
    default_checkpoint_interval = 50

    temp = default_temperature
    protocol = {'temperature':           [temp, temp, temp, temp, temp, temp], \
                'lambda_torsions':       [1.00, 0.80, 0.60, 0.40, 0.20, 0.00]}
    #------------------------------------
    

    input_prefix = options["input_prefix"]
    

    platform = get_fastest_platform()
    platform_name = platform.getName()
    if platform_name == "CUDA":
        # Set CUDA DeterministicForces (necessary for MBAR)
        platform.setPropertyDefaultValue('DeterministicForces', 'true')
        platform.setPropertyDefaultValue('Precision', 'mixed')
    else:
        #raise Exception("fastest platform is not CUDA")
        warnings.warn("fastest platform is not CUDA")
        #print("Fastest platform is {}".format(platorm_name), file=sys.stdout)
    

    # Deserialize system file and load system
    with open(os.path.join(input_prefix, 'system.xml'), 'r') as f:
        system = XmlSerializer.deserialize(f.read())

    # Deserialize integrator file and load integrator
    with open(os.path.join(input_prefix, 'integrator.xml'), 'r') as f:
        integrator = XmlSerializer.deserialize(f.read())

    # Set up simulation 
    pdb = PDBFile(os.path.join(input_prefix, 'state.pdb'))
    simulation = Simulation(pdb.topology, system, integrator, platform)

    # Load state
    with open(os.path.join(input_prefix, 'state.xml'), 'r') as f:
        state_xml = f.read()
    state = XmlSerializer.deserialize(state_xml)
    simulation.context.setState(state)



    # Calculate 6 torsion angles (α, β, γ, δ, ε, and ζ) around the consecutive chemical bonds, chi (χ) quantifying the relative base/sugar orientation
    # More details about RNA torsions can be found here: https://x3dna.org/highlights/pseudo-torsions-to-simplify-the-representation-of-dna-rna-backbone-conformation
    top = mdtraj.load_pdb(os.path.join(input_prefix, 'state.pdb'))
    n = Nucleic(topology=top.topology)
    idx, r =  n.get_bb_torsion_idx()
    idx = idx.reshape(28, 4)[3:]   # hard coded: (7 torsion/nb * 4 nb, 4 atoms/torsion)
    idx = np.array(idx)

    forces = list(system.getForces())
    torsion_indices = []
    for force in forces:
        name = force.__class__.__name__
        if "Torsion" in name:
            for i in range(force.getNumTorsions()):
                id1, id2, id3, id4, periodicity, phase, k = force.getTorsionParameters(i)
                #print(i, force.getTorsionParameters(i), file=stdout)
                x = np.array([id1, id2, id3, id4])
                for _idx in idx:
                    c = _idx == x
                    if c.all():
                        torsion_indices.append(i)
                        #print(i, force.getTorsionParameters(i))


    # Define alchemical region and thermodynamic state
    alchemical_region = alchemy.AlchemicalRegion(alchemical_torsions=torsion_indices)
    factory = alchemy.AbsoluteAlchemicalFactory()
    alchemical_system = factory.create_alchemical_system(system, alchemical_region)
    alchemical_state = alchemy.AlchemicalState.from_system(alchemical_system)
    thermodynamics_states = states.create_thermodynamic_state_protocol(alchemical_system, protocol=protocol, composable_states=[alchemical_state])
    sampler_states = states.SamplerState(positions=state.getPositions(), box_vectors=state.getPeriodicBoxVectors())


    # Rename old storge file
    storage_file = 'enhanced.nc'
    n = glob.glob(storage_file + '.nc')
    if os.path.exists(storage_file):
        #print('{} already exists. File will be renamed but checkpoint files will be deleted'.format(storage_file))
        os.remove('enhanced_checkpoint.nc')
        os.rename(storage_file, storage_file + "{}".format(str(len(n))))


    # Replica exchange sampler
    # LangevinSplittingDynamicsMove: High-quality Langevin integrator family based on symmetric Strang splittings, using g-BAOAB as default
    # BAOAB integrator (i.e. with V R O R V splitting), which was shown empirically to add a very small integration error in configurational space
    # https://github.com/openmm/openmm/issues/2520
    # https://openmmtools.readthedocs.io/en/stable/gettingstarted.html
    # https://openmmtools.readthedocs.io/en/stable/gettingstarted.html
    move =  mcmc.LangevinSplittingDynamicsMove(timestep=default_timestep, \
                                               n_steps=default_steps_per_replica, \
                                               collision_rate=default_collision_rate, \
                                               reassign_velocities=True, \
                                               splitting='V R O R V')  # default: "V R O R V"
    simulation = ReplicaExchangeSampler(mcmc_moves=move, number_of_iterations=default_number_of_iterations, replica_mixing_scheme=default_swap_scheme, online_analysis_interval=None)
    reporter = MultiStateReporter(storage='enhanced.nc', checkpoint_interval=default_checkpoint_interval)
    simulation.create(thermodynamics_states, sampler_states=sampler_states, storage=reporter)


    # Just to check if the performance is better using this - for Openmm <= 7.7
    from openmmtools.cache import ContextCache
    simulation.energy_context_cache = ContextCache(capacity=None, time_to_live=None, platform=platform)
    simulation.sampler_context_cache = ContextCache(capacity=None, time_to_live=None, platform=platform)


    # Run!
    simulation.run()



@click.command()
@click.option('--input_prefix', default='../eq', help='path to load xml files to create systems and pdb file to read topology')
def cli(**kwargs):
    run(**kwargs)



if __name__ == "__main__":
    cli()
