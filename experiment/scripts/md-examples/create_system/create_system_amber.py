#!/usr/bin/env python
# coding: utf-8

import os, sys, math
import glob as glob
import numpy as np
import re
import warnings
from openmm import *
from openmm.app import *
from openmm.unit import *
from pdbfixer import PDBFixer
from openff.toolkit.topology import Molecule, Topology
from openmmforcefields.generators import EspalomaTemplateGenerator, GAFFTemplateGenerator



inputfile = "../../crd/rna_noh.pdb"


#######################
# PAREMETERS
#######################
box_padding = 12.0 * angstrom
salt_conc = 0.15 * molar
nb_cutoff = 10 * angstrom
hmass = 3.5 * amu #Might need to be tuned to 3.5 amu 
water_model='tip3p'



"""
create system and minimize
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
integrator = LangevinMiddleIntegrator(300*kelvin, 1/picosecond, 0.002*picoseconds)
simulation = Simulation(modeller.topology, system, integrator)
simulation.context.setPositions(modeller.positions)

# minimize
simulation.minimizeEnergy()
minpositions = simulation.context.getState(getPositions=True).getPositions()
PDBFile.writeFile(modeller.topology, minpositions, open("min.pdb", 'w'))   


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
    xml = XmlSerializer.serialize(integrator)
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