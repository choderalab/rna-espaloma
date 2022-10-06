#!/usr/bin/env python

import os, sys
import numpy as np
import h5py
import click




def load_from_hdf5(kwargs):
    filename = kwargs["hdf5"]
    hdf = h5py.File(filename)
    gaff_version = kwargs["gaff"]

    for i, key in enumerate(list(hdf.keys())):
        record = hdf[key]
        if record["rna_type"][0].decode('UTF-8') == "trinucleotide":
            #nconfs = record["conformations"].shape[0]
            f = os.path.join("entries-{}".format(gaff_version), str(i), "mydata", "subtract-nonbonded")
            molfile = os.path.join(f, "mol.json")
            heterograph = os.path.join(f, "heterograph.bin")
            homograph = os.path.join(f, "homograph.bin")
            if os.path.exists(f) and os.path.exists(molfile) and os.path.exists(heterograph) and os.path.exists(homograph):
                pass
            else:
                print(i, key)   # index number is two less than the dl/XXXX.info because of the header and zero-indexing
                #del_dirpath = os.path.join("entries-{}".format(gaff_version), str(i))
                #shutil.rmtree(del_dirpath)
                


@click.command()
@click.option("--hdf5", default="dl/RNA-Single-Point-Dataset-v1.0.hdf5", help='hdf5 filename')
@click.option("--gaff", required=True, help="gaff version [gaff-1.81, gaff-2.11]")
def cli(**kwargs):
    #print(kwargs)
    load_from_hdf5(kwargs)



if __name__ == "__main__":
    cli()