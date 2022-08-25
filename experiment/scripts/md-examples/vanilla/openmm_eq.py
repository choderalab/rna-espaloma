#!/usr/bin/env python
# coding: utf-8

import os, sys, shutil
import pathlib
import glob as glob
import numpy as np
import re
import warnings
import mdtraj as md
import click
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



def create_position_restraint(position, restraint_atom_indices):
    """
    heavy atom restraint
    """
    force = CustomExternalForce("k*((x-x0)^2+(y-y0)^2+(z-z0)^2)")
    force.addGlobalParameter("k", 10.0*kilocalories_per_mole/angstroms**2)
    force.addPerParticleParameter("x0")
    force.addPerParticleParameter("y0")
    force.addPerParticleParameter("z0")
    for i in restraint_atom_indices:
        atom_crd = position[i]
        force.addParticle(i, atom_crd.value_in_unit(nanometers))
    return force



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
    inputfile = options["inputfile"]
    output_prefix = options["output_prefix"]
    box_padding = 12.0 * angstrom
    salt_conc = 0.15 * molar
    nb_cutoff = 10 * angstrom
    hmass = 3.5 * amu #Might need to be tuned to 3.5 amu 
    water_model='tip3p'
    timestep = 4 * femtoseconds
    hmass = 3.5 * amu
    temperature = 275 * kelvin
    pressure = 1 * atmosphere
    nsteps_min = 100
    nsteps_eq = 125000   # 50ps
    nsteps_prod = 2500000  # 10ns
    checkpoint_frequency = 250000  # 1ns
    logging_frequency = 25000  # 100ps
    netcdf_frequency = 25000  # 100ps

    # test
    #nsteps_min = 100
    #nsteps_eq = 100
    #nsteps_prod = 5000
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

    

    """
    create system
    """
    # pdbfixer: fix structure if necessary
    fixer = PDBFixer(filename=inputfile)
    fixer.findMissingResidues()
    fixer.findNonstandardResidues()
    fixer.replaceNonstandardResidues()
    fixer.removeHeterogens(True)
    fixer.findMissingAtoms()
    fixer.addMissingHydrogens(7.0)  # default: 7
    PDBFile.writeFile(fixer.topology, fixer.positions, open('pdbfixer.pdb', 'w'))

    # define force field
    ff = ForceField('amber/RNA.OL3.xml', 'amber/protein.ff14SB.xml', 'amber/tip3p_standard.xml', 'amber/tip3p_HFE_multivalent.xml')

    # solvate system
    modeller = Modeller(fixer.topology, fixer.positions)
    modeller.addSolvent(ff, model=water_model, padding=box_padding, ionicStrength=salt_conc)
    PDBFile.writeFile(modeller.topology, modeller.positions, file=open('solvated.pdb', 'w'))

    # create system
    system = ff.createSystem(modeller.topology, nonbondedMethod=PME, nonbondedCutoff=nb_cutoff, constraints=HBonds, rigidWater=True, hydrogenMass=hmass)
    integrator = LangevinMiddleIntegrator(temperature, 1/picosecond, timestep)
    simulation = Simulation(modeller.topology, system, integrator)
    simulation.context.setPositions(modeller.positions)


    # define reporter
    #simulation.reporters.append(PDBReporter('/Users/takabak/Desktop/dump.pdb', options["netcdf_frequency"]))
    simulation.reporters.append(NetCDFReporter(os.path.join(output_prefix, 'traj.nc'), netcdf_frequency))
    simulation.reporters.append(CheckpointReporter(os.path.join(output_prefix, 'checkpoint.chk'), checkpoint_frequency))
    simulation.reporters.append(StateDataReporter(os.path.join(output_prefix, 'reporter.log'), logging_frequency, step=True, potentialEnergy=True, kineticEnergy=True, totalEnergy=True, temperature=True, volume=True, density=True, speed=True))

    # minimization
    restraint_atom_indices = [ a.index for a in modeller.topology.atoms() if a.residue.name in ['A', 'C', 'U', 'T'] and a.element.symbol != 'H' ]
    restraint_index = system.addForce(create_position_restraint(modeller.positions, restraint_atom_indices))

    simulation.minimizeEnergy(maxIterations=nsteps_min)
    minpositions = simulation.context.getState(getPositions=True).getPositions()    
    PDBFile.writeFile(modeller.topology, minpositions, open(os.path.join(output_prefix, 'min.pdb'), 'w'))   


    # Equilibration
    # Heating
    n = 50
    for i in range(n):
        temp = temperature * i / n
        simulation.context.setVelocitiesToTemperature(temp)    # initialize velocity
        integrator.setTemperature(temp)    # set target temperature
        simulation.step(int(nsteps_eq/n))

    # NVT
    integrator.setTemperature(temperature)
    simulation.step(nsteps_eq)

    # NPT
    system.removeForce(restraint_index)
    system.addForce(MonteCarloBarostat(pressure, temperature))
    simulation.context.reinitialize(preserveState=True)
    simulation.step(nsteps_prod)

    """
    Export state in xml format
    """
    export_xml(simulation, system)



@click.command()
@click.option('--inputfile', '-i', required=True, help='path to input pdb file')
@click.option('--output_prefix', '-o', default=".", help='path to output files')
def cli(**kwargs):
    run(**kwargs)



if __name__ == "__main__":
    cli()