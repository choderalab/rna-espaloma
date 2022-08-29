#!/usr/bin/env python

import os, sys
import numpy as np
import torch
import espaloma as esp
import qcportal as ptl
import click
import pickle
from collections import Counter
from openff.toolkit.topology import Molecule
from openff.qcsubmit.results import BasicResultCollection
from simtk import unit
from simtk.unit import Quantity
from matplotlib import pyplot as plt



def get_graph(mol, energy, grad):
    offmol = Molecule.from_qcschema(mol, allow_undefined_stereo=True)
    offmol.compute_partial_charges_am1bcc()   # https://docs.openforcefield.org/projects/toolkit/en/0.9.2/api/generated/openff.toolkit.topology.Molecule.html
    charges = offmol.partial_charges.value_in_unit(esp.units.CHARGE_UNIT)
    g = esp.Graph(offmol)
    
    # energy is already hartree
    g.nodes["g"].data["u_ref"] = torch.tensor(
        [
            Quantity(
                energy,
                esp.units.HARTREE_PER_PARTICLE,
            ).value_in_unit(esp.units.ENERGY_UNIT)
        ],
        dtype=torch.get_default_dtype(),
    )[None, :]

    g.nodes["n1"].data["xyz"] = torch.tensor(
        np.stack(
            [
                Quantity(
                    mol.geometry,
                    unit.bohr,
                ).value_in_unit(esp.units.DISTANCE_UNIT)
            ],
            axis=1,
        ),
        requires_grad=True,
        dtype=torch.get_default_dtype(),
    )

    g.nodes["n1"].data["u_ref_prime"] = torch.stack(
        [
            torch.tensor(
                Quantity(
                    grad,
                    esp.units.HARTREE_PER_PARTICLE / unit.bohr,
                ).value_in_unit(esp.units.FORCE_UNIT),
                dtype=torch.get_default_dtype(),
            )
        ],
        dim=1,
    )
    
    g.nodes['n1'].data['q_hat'] = c = torch.tensor(charges, dtype=torch.get_default_dtype(),).unsqueeze(-1)
    
    return g, offmol





def load_from_qcarchive(kwargs):
    collection_type = kwargs["collection_type"]
    dataset_name = kwargs["dataset_name"]
    output_prefix = kwargs["output_prefix"]
    method = kwargs["method"]
    basis = kwargs["basis"]
    program = kwargs["program"]
    keywords = kwargs["keywords"]
    i = kwargs["entry_id"]


    collection_type = kwargs["collection_type"]
    if collection_type != "Dataset":
        raise NotImplementedError("{} not supported".format(collection_type))


    client = ptl.FractalClient()
    collection = client.get_collection(collection_type, dataset_name)
    recs = collection.get_records(method=method, basis=basis, program=program, keywords=keywords)

    #recs_d3bj = recs[0]
    #recs_b3lyp = recs[1]

    gs = []
    mol = client.query_molecules(recs[0].iloc[i].record.molecule)[0]
    if recs[0].iloc[i].record.status == 'COMPLETE' and recs[1].iloc[i].record.status == 'COMPLETE':
        print("#{}: {}".format(i, recs[0].iloc[i].name), file=sys.stdout)
        energy = recs[0].iloc[i].record.properties.return_energy
        energy += recs[1].iloc[i].record.properties.return_energy
        _grad1 = recs[0].iloc[i].record.return_result
        _grad2 = recs[1].iloc[i].record.return_result
        grad = _grad1 + _grad2

        g, offmol = get_graph(mol, energy, grad)
        gs.append(g)            
        ds = esp.data.dataset.GraphDataset(gs)
        ds.save(output_prefix)        
    else:
        offmol = Molecule.from_qcschema(mol, allow_undefined_stereo=True)
        offmol.compute_partial_charges_am1bcc()


    # save offmol with charges
    rid = recs[1].iloc[i].record.id
    with open(os.path.join(output_prefix, 'offmol_{}.pkl'.format(rid)), 'wb') as pkl:
        pickle.dump(offmol, pkl, protocol=4)



@click.command()
#@click.option("--collection_type", type=click.Choice=["Dataset", "ReactionDataset", "OptimizationDataset"], required=True, help='collection type used for QCArchive submission')
@click.option("--collection_type", default="Dataset", required=True, help='collection type used for QCArchive submission (currently only supports Dataset)')
@click.option("--dataset_name", default="RNA Single Point Dataset v1.0", required=True, help='name of the dataset')
@click.option("--output_prefix", default=".", required=True, help='output directory to save graph data')
@click.option("--method", default="b3lyp-d3bj", required=True, help="computational method to query")
@click.option("--basis", default="dzvp", required=True, help="computational basis query")
@click.option("--program", default="psi4", required=True, help="program to query on (default: psi4)")
@click.option("--keywords", default="default", help="option token desired")
@click.option("--entry_id", default=0, help="first entry id to start searching the data")
def cli(**kwargs):
    print(kwargs)
    load_from_qcarchive(kwargs)



if __name__ == "__main__":
    cli()
