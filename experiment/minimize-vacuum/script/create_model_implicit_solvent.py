#!/usr/bin/env python
# coding: utf-8

import os, sys, math
import numpy as np
import glob as glob
import mdtraj
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



def CreateAmberSystem(inputfile, forcefield_type, output_prefix):
    """
    """

    # pdbfixer
    fixer = PDBFixer(filename=inputfile)
    fixer.findMissingResidues()
    fixer.findNonstandardResidues()
    fixer.replaceNonstandardResidues()
    fixer.removeHeterogens(True)
    fixer.findMissingAtoms()
    fixer.addMissingHydrogens(7.0)  # default: 7
    PDBFile.writeFile(fixer.topology, fixer.positions, open('pdbfixer.pdb', 'w'))

    # create and check model
    amber_model = Modeller(fixer.topology, fixer.positions)

    logging.info("1. AMBER MODEL")
    logging.info("-------------------")
    for atom in amber_model.topology.atoms():
        logging.info("chainINDEX: {}, chainID: {}, resNAME: {:4s}, resID: {}, atomNAME: {:4s}, atomID: {}".format(atom.residue.chain.index, 
                                                                                                                  atom.residue.chain.id, \
                                                                                                                  atom.residue.name, \
                                                                                                                  atom.residue.id, \
                                                                                                                  atom.name, \
                                                                                                                  atom.id))

    # create system
    ff_amber = ForceField('amber/RNA.OL3.xml', 'implicit/obc2.xml')
    amber_system = ff_amber.createSystem(amber_model.topology)
    integrator = LangevinMiddleIntegrator(300*kelvin, 1/picosecond, 0.002*picoseconds)

    if forcefield_type == "amber":
        amber_simulation = Simulation(amber_model.topology, amber_system, integrator)
        amber_simulation.context.setPositions(amber_model.positions)

        print("before minimization: {}".format(amber_simulation.context.getState(getEnergy=True).getPotentialEnergy()))
        amber_simulation.minimizeEnergy(tolerance=5.0E-9, maxIterations=1500)
        print("after minimization: {}".format(amber_simulation.context.getState(getEnergy=True).getPotentialEnergy()))
        
        amber_minpositions = amber_simulation.context.getState(getPositions=True).getPositions()
        PDBFile.writeFile(amber_model.topology, amber_minpositions, open("{}_min_amber.pdb".format(output_prefix), 'w'))
        export_xml(amber_simulation, amber_system)
    else:
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

        # fix hydrogen positions
        restraint_index = amber_system.addForce(force)
        amber_simulation = Simulation(amber_model.topology, amber_system, integrator)
        amber_simulation.context.setPositions(amber_model.positions)
        amber_simulation.minimizeEnergy(maxIterations=10)
        amber_minpositions = amber_simulation.context.getState(getPositions=True).getPositions()
        # remove restraint
        amber_system.removeForce(restraint_index)


    # get state
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




def CreateEspalomaSystem(amber_model, amber_minpositions, amber_state, output_prefix):
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
    # save espaloma system
    topology = espaloma_model.getTopology()
    positions = espaloma_model.getPositions()
    PDBFile.writeFile(topology, positions, file=open('espaloma.pdb', 'w'))

    mol = Molecule.from_file('espaloma.pdb', file_format='pdb')
    generator = EspalomaTemplateGenerator(molecules=mol, forcefield='net.pt', reference_forcefield='openff_unconstrained-2.0.0', charge_method='nn')

    ff = ForceField('implicit/obc2.xml')
    ff.registerTemplateGenerator(generator.generator)
    system = ff.createSystem(espaloma_model.topology)

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
    # minimize 
    topology = espaloma_model_mapped.getTopology()
    positions = espaloma_model_mapped.getPositions()
    integrator = LangevinIntegrator(300*kelvin, 1/picosecond, 0.002*picoseconds)

    simulation = Simulation(espaloma_model_mapped.topology, system, integrator)
    simulation.context.setPositions(positions)

    print("before minimization: {}".format(simulation.context.getState(getEnergy=True).getPotentialEnergy()))
    simulation.minimizeEnergy(tolerance=5.0E-9, maxIterations=1500)
    print("after minimization: {}".format(simulation.context.getState(getEnergy=True).getPotentialEnergy()))
    
    minpositions = simulation.context.getState(getPositions=True).getPositions()
    PDBFile.writeFile(topology, minpositions, open("{}_min_espaloma.pdb".format(output_prefix), 'w')) 

    export_xml(simulation, system)

    return espaloma_model, espaloma_model_mapped



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




@click.command()
@click.option('--pdbfile', required=True, default='../../crd/rna_noh.pdb', help='path to pdb used to load topology')
@click.option('--forcefield_type', required=True, type=click.Choice(['amber', 'espaloma']))
@click.option('--output_prefix', help='output prefix')
def cli(**kwargs):
    inputfile = kwargs['pdbfile']
    forcefield_type = kwargs['forcefield_type']
    output_prefix = kwargs['output_prefix']

    # amber
    amber_model, amber_minpositions, amber_state = CreateAmberSystem(inputfile, forcefield_type, output_prefix)
    
    # espaloma
    if forcefield_type == "espaloma":
        espaloma_model, espaloma_model_mapped = CreateEspalomaSystem(amber_model, amber_minpositions, amber_state, output_prefix)

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
    First all residues will be into a single residue name. This is required to handle multiple residues with espaloma as 
    discussed [here](https://github.com/openmm/openmmforcefields/issues/228). Then we will convert it back to it's original residues to make our analysis easier.   
    
    Alternatively, we can use `openff.toolkit.topology` to convert the mdtraj topology into openmm topology as described in the 
    [openff-toolkit document](https://docs.openforcefield.org/projects/toolkit/en/0.9.2/api/generated/openff.toolkit.topology.Topology.html) to build an 
    espaloma system.

    Minimization level is set to tolerance=5E-9 and maximum iteration=1500 which is the same settings found from David Mobley's group.
    https://github.com/MobleyLab/off-ffcompare
    """
    cli()














