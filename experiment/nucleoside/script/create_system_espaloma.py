#!/usr/bin/env python
# coding: utf-8

import os, sys, math
import numpy as np
import glob as glob
import mdtraj
import copy
import re
import click
import warnings
from copy import deepcopy
from openmm import *
from openmm.app import *
from openmm.unit import *
from pdbfixer import PDBFixer
from openff.toolkit.topology import Molecule
from openff.toolkit.topology import Topology as OpenffTopology
from openmmforcefields.generators import EspalomaTemplateGenerator, GAFFTemplateGenerator
import logging
logging.basicConfig(filename='logging.log', encoding='utf-8', level=logging.INFO)


# Export version
import openmmforcefields
import openff.toolkit
print(f"openmmforcefield: {openmmforcefields.__version__}")
print(f"openff-toolkit: {openff.toolkit.__version__}")


#
# PAREMETERS and FORCE FIELD
#
box_padding = 12.0 * angstrom
salt_conc = 0.08 * molar
nb_cutoff = 10 * angstrom
hmass = 3.5 * amu #Might need to be tuned to 3.5 amu 
temperature = 300 * kelvin
timestep = 4 * femtoseconds


def CreateAmberSystem(inputfile, _ff, water_model):
    """
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
    
    # solvate system
    ff_amber = copy.deepcopy(_ff)
    amber_model = Modeller(fixer.topology, fixer.positions)
    amber_model.addSolvent(ff_amber, model=water_model, padding=box_padding, ionicStrength=salt_conc)
    PDBFile.writeFile(amber_model.topology, amber_model.positions, file=open('amber_solvated.pdb', 'w'))

    # check model
    logging.info("1. AMBER MODEL")
    logging.info("-------------------")
    for atom in amber_model.topology.atoms():
        logging.info("chainINDEX: {}, chainID: {}, resNAME: {:4s}, resID: {}, atomNAME: {:4s}, atomID: {}".format(atom.residue.chain.index, 
                                                                                                                  atom.residue.chain.id, \
                                                                                                                  atom.residue.name, \
                                                                                                                  atom.residue.id, \
                                                                                                                  atom.name, \
                                                                                                                  atom.id))

    # create system and apply heavy atom restraint
    amber_system = ff_amber.createSystem(amber_model.topology, nonbondedMethod=PME, nonbondedCutoff=nb_cutoff, constraints=HBonds, rigidWater=True, hydrogenMass=hmass)

    force = CustomExternalForce("k*((x-x0)^2+(y-y0)^2+(z-z0)^2)")
    force.addGlobalParameter("k", 100.0*kilocalories_per_mole/angstroms**2)
    force.addPerParticleParameter("x0")
    force.addPerParticleParameter("y0")
    force.addPerParticleParameter("z0")

    for i, atom in enumerate(amber_model.topology.atoms()):
        # atom symbols for tip4p models can be None
        try:
            if atom.element.symbol != "H":
                atom_crd = amber_model.positions[i]
                force.addParticle(i, atom_crd.value_in_unit(nanometers))
        except:
            pass
    amber_system.addForce(force)

    integrator = LangevinMiddleIntegrator(300*kelvin, 1/picosecond, 0.002*picoseconds)
    amber_simulation = Simulation(amber_model.topology, amber_system, integrator)
    amber_simulation.context.setPositions(amber_model.positions)

    # minimize: fix hydrogen positions
    amber_simulation.minimizeEnergy(maxIterations=10)
    amber_minpositions = amber_simulation.context.getState(getPositions=True).getPositions()
    PDBFile.writeFile(amber_model.topology, amber_minpositions, open("min.pdb", 'w'))   

    # get current state
    amber_state = amber_simulation.context.getState()

    return amber_model, amber_minpositions, amber_state



def create_new_topology(amber_model, amber_minpositions, amber_state):
    newTopology = Topology()
    newTopology.setPeriodicBoxVectors(amber_state.getPeriodicBoxVectors())
    newAtoms = {}
    newPositions = []*nanometer

    chain_counter = 0
    residue_counter = 0
    for chain in amber_model.topology.chains():
        if chain_counter == 0:
            chain_id = "A"
        elif chain_counter == 1:
            chain_id = "B"
        elif chain_counter == 2:
            chain_id = "C"
        elif chain_counter == 3:
            chain_id = "D"
        elif chain_counter == 4:
            chain_id = "E"
        
        newChain = newTopology.addChain(chain_id)
        #newChain = newTopology.addChain(chain.id)

        for residue in chain.residues():
            if chain.index == 0:
                # Merge all RNA residues into a single residue assuming the first chain is a RNA.
                if residue_counter == 0:
                    resname = 'RNA'
                    resid = '1'
                    newResidue = newTopology.addResidue(resname, newChain, resid, residue.insertionCode)
                    residue_counter += 1
                for atom in residue.atoms():
                    newAtom = newTopology.addAtom(atom.name, atom.element, newResidue, atom.id)
                    newAtoms[atom] = newAtom
                    newPositions.append(deepcopy(amber_minpositions[atom.index]))
            else:
                # Just copy the residue over.
                newResidue = newTopology.addResidue(residue.name, newChain, residue.id, residue.insertionCode)
                for atom in residue.atoms():
                    newAtom = newTopology.addAtom(atom.name, atom.element, newResidue, atom.id)
                    newAtoms[atom] = newAtom
                    newPositions.append(deepcopy(amber_minpositions[atom.index]))
                    
        chain_counter += 1
    for bond in amber_model.topology.bonds():
        if bond[0] in newAtoms and bond[1] in newAtoms:
            newTopology.addBond(newAtoms[bond[0]], newAtoms[bond[1]])

    # create espaloma model
    espaloma_model = copy.deepcopy(amber_model)
    espaloma_model.topology = newTopology
    espaloma_model.positions = newPositions

    return espaloma_model



def update_topology(amber_model, amber_state, espaloma_model):
    # convert espaloma topology to original topology. grab original information.
    atom_mapping = []
    for residue in amber_model.topology.residues():
        if residue.name == 'HOH': break

        #print(residue.name, residue.index, residue.id)
        a = [ {"name": atom.name, "index": atom.index, "resname": atom.residue.name, "resid": atom.residue.id } for atom in residue.atoms() ][0]
        atom_mapping.append(a)


    # update topology
    newTopology = Topology()
    newTopology.setPeriodicBoxVectors(amber_state.getPeriodicBoxVectors())
    newAtoms = {}
    newPositions = []*nanometer

    i = 0
    for chain in espaloma_model.topology.chains():    
        newChain = newTopology.addChain(chain.id)

        for residue in chain.residues():
            if residue.name == 'RNA':            
                for atom in residue.atoms():
                    try:
                        if atom_mapping[i]['name'] == atom.name and atom_mapping[i]['index'] == atom.index:
                            resname = atom_mapping[i]['resname']
                            resid = atom_mapping[i]['resid']
                            newResidue = newTopology.addResidue(resname, newChain, resid, residue.insertionCode)
                            i += 1
                    except:
                        pass

                    newAtom = newTopology.addAtom(atom.name, atom.element, newResidue, atom.id)
                    newAtoms[atom] = newAtom
                    newPositions.append(deepcopy(espaloma_model.positions[atom.index]))
            else:
                # Just copy the residue over.
                newResidue = newTopology.addResidue(residue.name, newChain, residue.id, residue.insertionCode)
                for atom in residue.atoms():
                    newAtom = newTopology.addAtom(atom.name, atom.element, newResidue, atom.id)
                    newAtoms[atom] = newAtom
                    newPositions.append(deepcopy(espaloma_model.positions[atom.index]))
                    
    for bond in espaloma_model.topology.bonds():
        if bond[0] in newAtoms and bond[1] in newAtoms:
            newTopology.addBond(newAtoms[bond[0]], newAtoms[bond[1]])

    # check updated system
    espaloma_model_mapped = copy.deepcopy(espaloma_model)
    espaloma_model_mapped.topology = newTopology
    espaloma_model_mapped.positions = newPositions

    return espaloma_model_mapped




def CreateEspalomaSystem(amber_model, amber_minpositions, amber_state, water_model, net_model):
    """
    Update topology and use minimized amber posisitons for new positions. 
    This is to ensure that the RNA structures are properly read by `openff.topology.Molecule`.
    """

    espaloma_model = create_new_topology(amber_model, amber_minpositions, amber_state)

    # check model
    logging.info("")
    logging.info("2. ESPALOMA MODEL")
    logging.info("-------------------")
    for atom in espaloma_model.topology.atoms():
        logging.info("chainINDEX: {}, chainID: {}, resNAME: {:4s}, resID: {}, atomNAME: {:4s}, atomID: {}".format(atom.residue.chain.index, 
                                                                                                                  atom.residue.chain.id, \
                                                                                                                  atom.residue.name, \
                                                                                                                  atom.residue.id, \
                                                                                                                  atom.name, \
                                                                                                                  atom.id))
    # save solvated espaloma system
    topology = espaloma_model.getTopology()
    positions = espaloma_model.getPositions()
    PDBFile.writeFile(topology, positions, file=open('espaloma_solvated.pdb', 'w'))

    # EspalomaTemplateGenerator to build Espaloma system
    t = mdtraj.load_pdb('espaloma_solvated.pdb')

    indices = t.topology.select('resname RNA')
    rna_topology = t.topology.subset(indices)
    rna_traj = t.atom_slice(indices)
    t.atom_slice(indices).save_pdb('rna_espaloma.pdb')

    mol = Molecule.from_file('rna_espaloma.pdb', file_format='pdb')
    generator = EspalomaTemplateGenerator(molecules=mol, forcefield=net_model, reference_forcefield='openff_unconstrained-2.0.0', charge_method='nn')
    #EspalomaTemplateGenerator.INSTALLED_FORCEFIELDS

    #ff = ForceField('amber/protein.ff14SB.xml', 'amber/tip3p_standard.xml', 'amber/tip3p_HFE_multivalent.xml')
    #ff = copy.deepcopy(_ff)
    if water_model == 'tip3p':
        ff = ForceField('amber/tip3p_standard.xml', 'amber/tip3p_HFE_multivalent.xml')
    elif water_model == 'tip3pfb':
        ff = ForceField('amber/tip3pfb_standard.xml', 'amber/tip3pfb_HFE_multivalent.xml')      
    elif water_model == 'spce':
        ff = ForceField('amber/spce_standard.xml', 'amber/spce_HFE_multivalent.xml')  
    elif water_model == 'opc3':
        ff = ForceField('amber/opc3_standard.xml')
    elif water_model == 'tip4pew':
        ff = ForceField('amber/tip4pew_standard.xml', 'amber/tip4pew_HFE_multivalent.xml')
    elif water_model == 'tip4pfb':
        ff = ForceField('amber/tip4pfb_standard.xml', 'amber/tip4pfb_HFE_multivalent.xml')
    elif water_model == 'opc':
        ff = ForceField('amber/opc_standard.xml')
    else:
        raise NameError("undefined water model")
    ff.registerTemplateGenerator(generator.generator)
    system = ff.createSystem(espaloma_model.topology, nonbondedMethod=PME, nonbondedCutoff=nb_cutoff, constraints=HBonds, rigidWater=True, hydrogenMass=hmass)

    # update topology (rename residues to it's original names)
    espaloma_model_mapped = update_topology(amber_model, amber_state, espaloma_model)

    # check model
    logging.info("")
    logging.info("3. ESPALOMA MAPPED MODEL")
    logging.info("-------------------")
    for atom in espaloma_model_mapped.topology.atoms():
        logging.info("chainINDEX: {}, chainID: {}, resNAME: {:4s}, resID: {}, atomNAME: {:4s}, atomID: {}".format(atom.residue.chain.index, 
                                                                                                                  atom.residue.chain.id, \
                                                                                                                  atom.residue.name, \
                                                                                                                  atom.residue.id, \
                                                                                                                  atom.name, \
                                                                                                                  atom.id))
    topology = espaloma_model_mapped.getTopology()
    positions = espaloma_model_mapped.getPositions()
    PDBFile.writeFile(topology, positions, file=open('espaloma_mapped_solvated.pdb', 'w'))

    #integrator = LangevinIntegrator(300*kelvin, 1/picosecond, 0.002*picoseconds)
    integrator = LangevinMiddleIntegrator(temperature, 1/picosecond, timestep)
    simulation = Simulation(espaloma_model_mapped.topology, system, integrator)
    simulation.context.setPositions(positions)
    
    # minimize 
    #simulation.minimizeEnergy(maxIterations=100)
    #minpositions = simulation.context.getState(getPositions=True).getPositions()
    #PDBFile.writeFile(topology, minpositions, open("espaloma_mapped_min.pdb", 'w')) 

    # save
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
                #enforcePeriodicBox=True).getPositions(),
                enforcePeriodicBox=False).getPositions(),
                file=wf,
                keepIds=True
        )

    # Save and serialize system
    system.setDefaultPeriodicBoxVectors(*state.getPeriodicBoxVectors())
    with open("system.xml", "w") as wf:
        xml = XmlSerializer.serialize(system)
        wf.write(xml)


    return espaloma_model, espaloma_model_mapped




@click.command()
@click.option('--pdbfile', required=True, default='../../../crd/rna_noh.pdb', help='path to pdb used to load topology')
@click.option('--water_model', type=click.Choice(['tip3p', 'tip3pfb', 'spce', 'opc3', 'tip4pew', 'tip4pfb', 'opc']), help='water model')
@click.option('--net_model', required=True, help='path to espaloma model')
def cli(**kwargs):
    inputfile = kwargs['pdbfile']
    water_model = kwargs['water_model']
    net_model = kwargs['net_model']

    #
    # 3 point water model
    #
    if water_model == 'tip3p':
        water_model='tip3p'
        _ff = ForceField('amber/RNA.OL3.xml', 'amber/protein.ff14SB.xml', 'amber/tip3p_standard.xml', 'amber/tip3p_HFE_multivalent.xml')
    elif water_model == 'tip3pfb':
        water_model='tip3p'
        _ff = ForceField('amber/RNA.OL3.xml', 'amber/protein.ff14SB.xml', 'amber/tip3pfb_standard.xml', 'amber/tip3pfb_HFE_multivalent.xml')      
    elif water_model == 'spce':
        water_model='tip3p'
        _ff = ForceField('amber/RNA.OL3.xml', 'amber/protein.ff14SB.xml', 'amber/spce_standard.xml', 'amber/spce_HFE_multivalent.xml')  
    elif water_model == 'opc3':
        # OPC3: Exclude residue U from ions/ionslm_126_opc3.xml. ValueError will raise because of same residue template name.
        #       (ValueError: Residue template U with the same override level 0 already exists.)
        #       `ionslm_126_opc3.xml` manually modified in
        #       /home/takabak/mambaforge/envs/openmmforcefields-dev/lib/python3.9/site-packages/openmmforcefields-0.11.0+64.gd78f2b8.dirty-py3.9.egg/openmmforcefields/ffxml/amber/ions
        water_model='tip3p'
        _ff = ForceField('amber/RNA.OL3.xml', 'amber/protein.ff14SB.xml', 'amber/opc3_standard.xml')
    
    #
    # 4 point water model
    #
    elif water_model == 'tip4pew':
        water_model='tip4pew'
        _ff = ForceField('amber/RNA.OL3.xml', 'amber/protein.ff14SB.xml', 'amber/tip4pew_standard.xml', 'amber/tip4pew_HFE_multivalent.xml')
    elif water_model == 'tip4pfb':
        water_model='tip4pew'
        _ff = ForceField('amber/RNA.OL3.xml', 'amber/protein.ff14SB.xml', 'amber/tip4pfb_standard.xml', 'amber/tip4pfb_HFE_multivalent.xml')
    elif water_model == 'opc':
        water_model='tip4pew'
        _ff = ForceField('amber/RNA.OL3.xml', 'amber/protein.ff14SB.xml', 'amber/opc_standard.xml')

    else:
        raise NameError("undefined water model")

    amber_model, amber_minpositions, amber_state = CreateAmberSystem(inputfile, _ff, water_model)
    espaloma_model, espaloma_model_mapped = CreateEspalomaSystem(amber_model, amber_minpositions, amber_state, water_model, net_model)

    # compare models
    logging.info("")
    logging.info("4. COMPARE MODELS")
    logging.info("-------------------")
    logging.info("AMBER:           {}".format(amber_model.topology))
    logging.info("ESPALOMA:        {}".format(espaloma_model.topology))
    logging.info("ESPALOMA MAPPED: {}".format(espaloma_model_mapped.topology))


    


if __name__ == "__main__":
    """
    Building espaloma system
    ------------

    We will use the `EspalomaTemplateGenerator` to build the espaloma system. Details can be found [here](https://github.com/openmm/openmmforcefields).

    First we will solvate and add ions to the system with amber force field. Then we will use the solvated amber system to generate the espaloma model.
    Missing hydrogen atoms and box waters will be added with PDBFixer and Modeller function, respectively. Note that PDBFixer sometimes adds the missing 
    hydrogen atoms in a weird position (e.g. nucleotide bases). The solvated amber system will be minimized to ensure that all hydrogens added 
    by PDBFixer are properly placed. 

    We will rename the residues which we want to assign the espaloma force field, in this case all RNA residues, into a single residue. This is required 
    to handle multiple residues with espaloma as discussed [here](https://github.com/openmm/openmmforcefields/issues/228). Then we will convert it back 
    to it's original residues to make our analysis easier.   
    
    Alternatively, we can use `openff.toolkit.topology` to convert the mdtraj topology into openmm topology as described in the 
    [openff-toolkit document](https://docs.openforcefield.org/projects/toolkit/en/0.9.2/api/generated/openff.toolkit.topology.Topology.html) to build an 
    espaloma system.
    """
    cli()














