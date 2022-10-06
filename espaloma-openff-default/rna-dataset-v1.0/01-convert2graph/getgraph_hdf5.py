#!/usr/bin/env python

import os, sys
import numpy as np
import h5py
import torch
import espaloma as esp
from espaloma.units import *
from espaloma.data.md import *
import qcportal as ptl
import click
from collections import Counter
from openff.toolkit.topology import Molecule
from openff.qcsubmit.results import BasicResultCollection
from simtk import unit
from simtk.unit import Quantity
from matplotlib import pyplot as plt



def get_graph(record):
    smi = record["smiles"][0].decode('UTF-8')
    offmol = Molecule.from_smiles(smi, allow_undefined_stereo=True)
    offmol.compute_partial_charges_am1bcc()   # https://docs.openforcefield.org/projects/toolkit/en/0.9.2/api/generated/openff.toolkit.topology.Molecule.html
    charges = offmol.partial_charges.value_in_unit(esp.units.CHARGE_UNIT)
    g = esp.Graph(offmol)

    # energy
    energy = []
    for e, e_corr in zip(record["dft_total_energy"], record["dispersion_correction_energy"]):
        energy.append(e + e_corr)
    
    # gradient
    grad = []
    for gr, gr_corr in zip(record["dft_total_gradient"], record["dispersion_correction_gradient"]):
        grad.append(gr + gr_corr)
        
    # conformations
    conformations = record["conformations"]

    # energy is already hartree
    g.nodes["g"].data["u_ref"] = torch.tensor(
        [
            Quantity(
                _energy,
                esp.units.HARTREE_PER_PARTICLE,
            ).value_in_unit(esp.units.ENERGY_UNIT)
            for _energy in energy
        ],
        dtype=torch.get_default_dtype(),
    )[None, :]

    g.nodes["n1"].data["xyz"] = torch.tensor(
        np.stack(
            [
                Quantity(
                    xyz,
                    unit.bohr,
                ).value_in_unit(esp.units.DISTANCE_UNIT)
                for xyz in conformations
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
                    _grad,
                    esp.units.HARTREE_PER_PARTICLE / unit.bohr,
                ).value_in_unit(esp.units.FORCE_UNIT),
                dtype=torch.get_default_dtype(),
            )
            for _grad in grad
        ],
        dim=1,
    )
    
    g.nodes['n1'].data['q_ref'] = c = torch.tensor(charges, dtype=torch.get_default_dtype(),).unsqueeze(-1)
    
    return g



def load_from_hdf5(kwargs):
    filename = kwargs["hdf5"]
    key = kwargs["keyname"]
    output_prefix = kwargs["output_prefix"]
    gaff_version = kwargs["gaff"]

    hdf = h5py.File(filename)
    record = hdf[key]

    #gs = []
    #g = get_graph(record)
    #gs.append(g)
    #ds = esp.data.dataset.GraphDataset(gs)
    #ds.save(output_prefix)
    g = get_graph(record)
    g.save(output_prefix)


    # subtract nonbonded interaction energy
    #e1 = g.nodes['g'].data['u_ref'].item()
    e1 = [ u.item() for u in g.nodes['g'].data['u_ref'][0] ]
    g = subtract_nonbonded_force(g, forcefield=gaff_version, subtract_charges=True)   # default: forcefield=gaff-1.81
    #e2 = g.nodes['g'].data['u_ref'].item()
    e2 = [ u.item() for u in g.nodes['g'].data['u_ref'][0] ]
    print("{} (before) / {} (after)".format(e1, e2), file=sys.stdout)
    g.save(os.path.join(output_prefix, "subtract-nonbonded"))


@click.command()
@click.option("--hdf5", required=True, help='hdf5 filename')
@click.option("--keyname", required=True, help='keyname of the hdf5 group')
@click.option("--output_prefix", required=True, help='output directory to save graph data')
#@click.option("--gaff", default="gaff-1.81", help="gaff version (default: gaff-1.81")
@click.option("--gaff", required=True, help="gaff version [gaff-1.81, gaff-2.11]")
def cli(**kwargs):
    print(kwargs)
    load_from_hdf5(kwargs)



if __name__ == "__main__":
    cli()
