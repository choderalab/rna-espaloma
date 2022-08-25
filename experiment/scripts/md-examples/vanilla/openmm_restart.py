#!/usr/bin/env python
# coding: utf-8

import os, sys, shutil
import pathlib
import glob as glob
import numpy as np
import re
import warnings
import click
import mdtraj as md
import yaml
import openmmtools as mmtools
from openmm.app import *
from openmm import *
from openmm.unit import *
#from simtk.unit import Quantity
from openff.toolkit.utils import utils as offutils
#from openff.units.openmm import to_openmm
from sys import stdout
from openmm.app import PDBFile
from pdbfixer import PDBFixer
from mdtraj.reporters import NetCDFReporter



def export_xml(simulation, system):
    """
    Save state as XML
    """
    state = simulation.context.getState(getPositions=True, getVelocities=True, getEnergy=True, getForces=True)
    # Save and serialize the final state
    with open("state.xml", "w") as wf:
        xml = XmlSerializer.serialize(state)
        wf.write(xml)

    # Save and serialize integrator
    with open("integrator.xml", "w") as wf:
        xml = XmlSerializer.serialize(simulation.integrator)
        wf.write(xml)

    # Save the final state as a PDB
    with open("state.pdb", "w") as wf:
        PDBFile.writeFile(
            simulation.topology,
            simulation.context.getState(
                getPositions=True,
                enforcePeriodicBox=True).getPositions(),
                file=wf,
                keepIds=True
        )

    # Save and serialize system
    system.setDefaultPeriodicBoxVectors(*state.getPeriodicBoxVectors())
    with open("system.xml", "w") as wf:
        xml = XmlSerializer.serialize(system)
        wf.write(xml)



def run(**options):
    #print(options)
    pdbfile = options["pdbfile"]
    restart_prefix = options["restart_prefix"]
    initialize_velocity = options["initialize_velocity"]
    timestep = 4 * femtoseconds
    hmass = 3.5 * amu
    temperature = 275 * kelvin
    checkpoint_frequency = 250000  # 1ns
    logging_frequency = 25000  # 100ps
    netcdf_frequency = 25000  # 100ps
    nsteps = 25000000  # 100ns
    
    # test
    #nsteps = 5000
    #checkpoint_frequency = 10
    #logging_frequency = 10
    #netcdf_frequency = 10


    platform = mmtools.utils.get_fastest_platform()
    platform_name = platform.getName()
    print("fastest platform is ", platform_name)
    if platform_name == "CUDA":
        # Set CUDA DeterministicForces (necessary for MBAR)
        platform.setPropertyDefaultValue('DeterministicForces', 'true')
        platform.setPropertyDefaultValue('Precision', 'mixed')
    else:
        #raise Exception("fastest platform is not CUDA")
        warnings.warn("fastest platform is not CUDA")


    # Deserialize system file and load system
    with open(os.path.join(restart_prefix, 'system.xml'), 'r') as f:
        system = XmlSerializer.deserialize(f.read())

    # Deserialize integrator file and load integrator
    with open(os.path.join(restart_prefix, 'integrator.xml'), 'r') as f:
        integrator = XmlSerializer.deserialize(f.read())

    # Set up simulation 
    pdb = PDBFile(pdbfile)
    simulation = Simulation(pdb.topology, system, integrator, platform)

    # Load state
    with open(os.path.join(restart_prefix, 'state.xml'), 'r') as f:
        state_xml = f.read()
    state = XmlSerializer.deserialize(state_xml)
    simulation.context.setState(state)


    # Define reporter
    #simulation.reporters.append(PDBReporter('/Users/takabak/Desktop/dump.pdb', options["netcdf_frequency"]))
    simulation.reporters.append(NetCDFReporter('traj.nc', netcdf_frequency))
    simulation.reporters.append(CheckpointReporter('checkpoint.chk', checkpoint_frequency))
    simulation.reporters.append(StateDataReporter('reporter.log', logging_frequency, step=True, potentialEnergy=True, kineticEnergy=True, totalEnergy=True, temperature=True, volume=True, density=True, speed=True))

    if initialize_velocity == "True":
        simulation.context.setVelocitiesToTemperature(temperature)     # initialize velocity
    simulation.step(nsteps)


    """
    Export state in xml format
    """
    export_xml(simulation, system)
 


@click.command()
@click.option('--pdbfile', required=True, default='../md0/state.pdb', help='path to pdb used to load topology')
@click.option('--restart_prefix', default='.', help='path to load restart files')
@click.option('--initialize_velocity', is_flag=False, help='initialize velocity')
def cli(**kwargs):
    run(**kwargs)



if __name__ == "__main__":
    cli()