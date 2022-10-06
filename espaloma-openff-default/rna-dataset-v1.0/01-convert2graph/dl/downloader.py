#!/usr/bin/env python
# coding: utf-8
from qcportal import FractalClient
from collections import defaultdict
from rdkit import Chem
import numpy as np
import h5py
import yaml



# Units for a variety of fields that can be downloaded.
units = {'dft_total_energy': 'hartree',
         'dft_total_gradient': 'hartree/bohr',
         'dispersion_total_energy': 'hartree',
         'dispersion_total_gradient': 'hartree/bohr',
         'mbis_charges': 'elementary_charge',
         'mbis_dipoles': 'elementary_charge*bohr',
         'mbis_quadrupoles': 'elementary_charge*bohr^2',
         'mbis_octupoles': 'elementary_charge*bohr^3',
         'scf_dipole': 'elementary_charge*bohr',
         'scf_quadrupole': 'elementary_charge*bohr^2'}



# Process the configuration file and download data.
with open('config.yaml') as input:
    config = yaml.safe_load(input.read())
if 'max_force' in config:
    max_force = float(config['max_force'])
else:
    max_force = None
client = FractalClient()
outputfile = h5py.File('RNA-Single-Point-Dataset-v1.0.hdf5', 'w')



for subset in config['subsets']:
    ds = client.get_collection('Dataset', subset)
    all_molecules = ds.get_molecules()
    #spec = ds.list_records().iloc[0].to_dict()
    spec = ds.list_records().iloc[-1].to_dict()
    assert spec['method'] == 'b3lyp-d3bj'



    # QCFractal divides the B3LYP-D3BJ calculation into two separate parts; functional b3lyp evaluation, and dispersion d3bj evaluation
    # index=0 in this refers to the d3bj calculation and index=1 is the b3lyp calculation
    # properties from respective record dictionaries can be added to get the final property, for example final energy of b3lyp-d3bj calculation
    recs = ds.get_records(method=spec['method'], basis=spec['basis'], program=spec['program'], keywords=spec['keywords'])
    #print(spec['method'], spec['basis'], spec['program'], spec['keywords'])

    recs_by_name = defaultdict(list)
    mols_by_name = defaultdict(list)
    for i in range(len(recs[0])):
        rec_d3bj  = recs[0].iloc[i].record
        rec_b3lyp = recs[1].iloc[i].record
        if rec_d3bj is not None and rec_b3lyp is not None and rec_d3bj.status == 'COMPLETE' and rec_b3lyp.status == 'COMPLETE':
            assert recs[0].index[i] == recs[1].index[i], print("#{}: index does not match".format(i))
            index = recs[0].index[i]
            name = index[:index.rfind('-')]
            #print(recs[1].iloc[i].name, "\t", name)
            recs_by_name[name].append([rec_d3bj, rec_b3lyp])
            mols_by_name[name].append(all_molecules.loc[index][0])



    # check if number of unique molecules matches original qca submission
    # https://github.com/openforcefield/qca-dataset-submission/tree/master/submissions/2022-07-07-RNA-basepair-triplebase-single-points
    n_mols = 94
    assert len(recs_by_name) == n_mols, print("number of unique molecules in record ({}) does not match number of qcarchive submission entries ({})".format(len(recs_by_name), n_mols))



    # Add the data to the HDF5 file.
    mydict = { "1": "trinucleotide", "2": "base pair", "3": "base triple" }

    for name in recs_by_name:
        #print(name)
        group_recs = recs_by_name[name]
        molecules = mols_by_name[name]
        #qcvars = [r.extras['qcvars'] for r in group_recs]
        qcvars_d3bj  = [r[0].extras['qcvars'] for r in group_recs]
        qcvars_b3lyp = [r[1].extras['qcvars'] for r in group_recs]
        smiles = molecules[0].extras['canonical_isomeric_explicit_hydrogen_mapped_smiles']
        #ref_energy = compute_reference_energy(smiles)
        name = name.replace('/', '')  # Remove stereochemistry markers that h5py interprets as path separators
        
        group = outputfile.create_group(name)
        group.create_dataset('subset', data=[subset], dtype=h5py.string_dtype())
        group.create_dataset('smiles', data=[smiles], dtype=h5py.string_dtype())
        group.create_dataset("atomic_numbers", data=molecules[0].atomic_numbers, dtype=np.int16)
        
        # rna type
        n = name.split('.')
        group.create_dataset('rna_type', data=[mydict[str(len(n))]], dtype=h5py.string_dtype())
        
        if max_force is not None:
            force = np.array([vars['DFT TOTAL GRADIENT'] for vars in qcvars])
            samples = [i for i in range(len(molecules)) if np.max(np.abs(force[i])) <= max_force]
            molecules = [molecules[i] for i in samples]
            qcvars = [qcvars[i] for i in samples]

        # conformations
        ds = group.create_dataset('conformations', data=np.array([m.geometry for m in molecules]), dtype=np.float32)
        ds.attrs['units'] = 'bohr'
        
        # formation energy
        #ds = group.create_dataset('formation_energy', data=np.array([vars['DFT TOTAL ENERGY']-ref_energy for vars in qcvars]), dtype=np.float32)
        #ds.attrs['units'] = 'hartree'

        for value in config['values']:
            key = value.lower().replace(' ', '_')
            try:
                if key.startswith("dispersion"):
                    qcvars = qcvars_d3bj
                else:
                    qcvars = qcvars_b3lyp
                ds = group.create_dataset(key, data=np.array([v[value] for v in qcvars], dtype=np.float32), dtype=np.float32)
                if key in units:
                    ds.attrs['units'] = units[key]
            except KeyError:
                pass


    # export basic information
    output_basename = subset.replace(' ', '-')
    with open("{}.info".format(output_basename), "w") as wf:
        wf.write('RECORD_NAME\tNUMBER_OF_CONFORMATIONS\tRNA_TYPE\tMAPPED_SMILES\n')
        for name in recs_by_name:
            molecules = mols_by_name[name]
            smiles = molecules[0].extras['canonical_isomeric_explicit_hydrogen_mapped_smiles']
            n = name.split('.')
            wf.write('{}\t{}\t{}\t{}\n'.format(name, len(molecules), mydict[str(len(n))], smiles))