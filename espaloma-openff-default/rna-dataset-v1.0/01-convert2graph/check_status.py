#!/usr/bin/env python

import os, sys
import numpy as np
import qcportal as ptl
import click



def load_from_qcarchive(kwargs):
    collection_type = kwargs["collection_type"]
    dataset_name = kwargs["dataset_name"]
    output_prefix = kwargs["output_prefix"]
    method = kwargs["method"]
    basis = kwargs["basis"]
    program = kwargs["program"]
    keywords = kwargs["keywords"]


    collection_type = kwargs["collection_type"]
    if collection_type != "Dataset":
        raise NotImplementedError("{} not supported".format(collection_type))


    client = ptl.FractalClient()
    collection = client.get_collection(collection_type, dataset_name)
    recs = collection.get_records(method=method, basis=basis, program=program, keywords=keywords)

    with open("mylist", "w") as wf:
        for i in range(len(recs[0])):
            rid = recs[0].iloc[i].record.id
            smi = recs[0].iloc[i].name
            nb = [ s for s in smi if s == "." ]    # check if the entry is a trinucleotide, base pair, or base triple based on the number of "." found in the smiles string

            if recs[0].iloc[i].record.status == 'COMPLETE' and recs[1].iloc[i].record.status == 'COMPLETE':
                f = os.path.join("entries", str(i), "mydata", "0")
                molfile = os.path.join(f, "mol.json")
                heterograph = os.path.join(f, "heterograph.bin")
                homograph = os.path.join(f, "homograph.bin")

                if os.path.exists(f) and os.path.exists(molfile) and os.path.exists(heterograph) and os.path.exists(homograph):
                    print("{:4d} \t status:COMPLETE   \t calc:SUCCESS \t # of mols in record:{}".format(i, len(nb)+1))
                else:
                    print("{:4d} \t status:COMPLETE   \t calc:FAILED  \t # of mols in record:{}".format(i, len(nb)+1))
                    # write entry id to redo calculation
                    #if len(nb) + 1 == 1:
                    #    wf.write("{}\n".format(str(i)))
                    wf.write("{}\n".format(str(i)))
            else:
                print("{:4d} \t status:INCOMPLETE \t calc:INVALID \t # of mols in record:{}".format(i, len(nb)+1))


@click.command()
@click.option("--collection_type", default="Dataset", help='collection type used for QCArchive submission (currently only supports Dataset)')
@click.option("--dataset_name", default="RNA Single Point Dataset v1.0", help='name of the dataset')
@click.option("--output_prefix", default="mydata", help='output directory to save graph data')
@click.option("--method", default="b3lyp-d3bj", required=True, help="computational method to query")
@click.option("--basis", default="dzvp", required=True, help="computational basis query")
@click.option("--program", default="psi4", help="program to query on (default: psi4)")
@click.option("--keywords", default="default", help="option token desired")
def cli(**kwargs):
    #print(kwargs)
    load_from_qcarchive(kwargs)


if __name__ == "__main__":
    cli()