#!/usr/bin/env python
# coding: utf-8

"""
Reweight RNA tetramer conformer population with PyMBAR
=======================================

`pymbar.timeseries.detectEqulibration` gives slightly different t0, g, and Neff_max than `openmmtools`.
This is expceted because in openmmtools `mmtools.multistate.MultiStateSamplerAnalyzer()._get_equilibraiton_data` uses a modified pass-through of `pymbar.timeseries.detectEquilibration`.
More details can be found by running `mmtools.multistate.utils.get_equilibration_data_per_sample?`
"""

import os, sys, math
import numpy as np
import click
import glob
import mdtraj
import logging
import netCDF4 as nc
import warnings
import pandas as pd
from collections import defaultdict
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, FormatStrFormatter, AutoMinorLocator
import seaborn as sns
import openmmtools as mmtools
import barnaba as bb
from pymbar import MBAR
from barnaba import definitions
from barnaba.nucleic import Nucleic


# ==============================================================================
# GLOBAL PARAMETERS
# ==============================================================================

# warnings
warnings.filterwarnings("ignore")

# logging
logger = logging.getLogger(__name__)
logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)

# pandas
pd.options.display.max_rows = None
pd.options.display.max_columns = None
pd.options.display.precision = 1
pd.options.display.float_format = '{:.1f}'.format

params_mydefault = {'legend.fontsize': 40, 
                    'font.size': 40, 
                    'axes.labelsize': 48,
                    'axes.titlesize': 48,
                    'xtick.labelsize': 40,
                    'ytick.labelsize': 40,
                    'savefig.dpi': 600, 
                    'figure.figsize': [64, 8],
                    'xtick.major.size': 10,
                    'xtick.minor.size': 7,
                    'ytick.major.size': 10,
                    'ytick.minor.size': 7}
plt.rcParams.update(params_mydefault)

# structure annotation
myclass_mapping_dict = {"AMa": 1, "AMi":2, "I": 3, "F1": 4, "F4": 5, "O": 6}

# define color
mycolor_dict = { "AMa": "green", "AMi": "blue", "I": "red", "F1": "magenta", "F4": "orange", "O": "black" }

# settings
UNIT_NM_TO_ANGSTROMS = 10
UNIT_PS_TO_NS = 1/1000



# ==============================================================================
# SUBROUTINE
# ==============================================================================

def radian_to_degree(a):
    """
    Convert radian to degree.
    
    Parameters
    ----------
    a : list or np.ndarray
        angles in radians of shape [n_frames, n_residues, n_torsions]
    
    Returns
    -------
    a : list or np.ndarray
        angles in degrees of shape [n_frames, n_residues, n_torsions]
    """

    a[np.where(a<0.0)] += 2.*np.pi
    a *= 180.0/np.pi
    #a = a*(180./np.pi)
    #a[np.where(a<0.0)] += 360
    
    return a


def plot(x, y, yerr, plot_title, output_prefix):
    """
    Plot conformation population.

    Paramters
    ---------
    x : list or np.ndarray
        Cateogry names.
    y : list or np.ndarray
        Population of each category.
    yerr : list or np.ndarray or None
        Population error used for histogram error bars.
    
    Returns
    -------
    """
    
    fig, ax = plt.subplots(figsize=(16, 8))
    ax.set_ylabel('(%)')
    ax.yaxis.set_label_position("left")
    ax.yaxis.set_minor_locator(MultipleLocator(10))
    ax.yaxis.set_ticks_position("left")
    ax.set_ylim([0, 100])

    for i, _y in enumerate(y):
        ax.text(x=i-0.2, y=_y+3, s=f'{_y:.1f}', size=24)

    #ax.bar(myclass_mbar.keys(), population, width=1.0, color=mycolor_dict.values())
    if yerr != None:
        ax.bar(x, y, width=1.0, color=mycolor_dict.values())
    else:
        ax.bar(x, y, width=1.0, color=mycolor_dict.values(), yerr=yerr)
    plt.tight_layout()
    if plot_title != None:
        plt.title(plot_title)
    plt.subplots_adjust(hspace=0.1, wspace=0.1)
    plt.savefig("{}/population.png".format(output_prefix))


def extract_data(replica_index, bb_angles, pucker_angles, stackings):
    """
    Extract pre-calculated properties.

    Parameters
    ----------
    replica_index : int
        Replica index number
    bb_angles : np.ndarray of shape [n_replicas, n_iterations, n_residues, n_angles]
        Backbone angles in degrees
    pucker_angles : np.ndarray of shape [n_replicas, n_iterations, n_residues, n_angles]
        Pucker angles in degrees
    stackings : np.ndarray of shape [n_replicas, n_iterations, stacking_info]
        Stacking_info stores stacking residue index and stacking pattern (e.g. [[[0, 1], [1, 2], [2, 3]], ['>>', '>>', '>>']])

    Returns
    -------
    """
    alpha = bb_angles[replica_index,:,:,0]  # [n_iterations, n_residues]
    beta = bb_angles[replica_index,:,:,1]
    gamma = bb_angles[replica_index,:,:,2]
    delta = bb_angles[replica_index,:,:,3]
    eps = bb_angles[replica_index,:,:,4]
    zeta = bb_angles[replica_index,:,:,5]
    chi = bb_angles[replica_index,:,:,6]
    phase = pucker_angles[replica_index,:,:,0]
    stacking = stackings[replica_index,:,:]
    
    return alpha, beta, gamma, delta, eps, zeta, chi, phase, stacking



# ==============================================================================
# STRUCTURE ANNOTATION
# ==============================================================================

def _check_endo(angle_d, angle_p):
    """
    Define C3'-endo and C2'-endo.

    δ torsion angles is used to defined the endo states described in:
    RNA backbone: Consensus all-angle conformers and modular string nomenclature (an RNA Ontology Consortium contribution), RNA 2008, doi: 10.1261/rna.657708

    C3'-endo:
        an individual ribose with δ between 55° and 110°
    C2'-endo:
        an individual ribose with δ between 120° and 175°
        
    Alternatively C3'- and C2'- endo can be defined using the pucker phase angle. C3'-endo [0°, 36°) as in canonical RNA and A-form DNA, and the C2'-endo [144°, 180°).

    Returns
    -------
    c3_endo : list
        '1' if if δ torsion angle forms a C3'-endo form, else '0'

    c2_endo : list
        '1' if if δ torsion angle forms a C2'-endo form, else '0'
    """

    c3_endo = []
    for _delta, _phase in zip(angle_d, angle_p):
        # C3 endo
        if (_delta >= 55 and _delta < 110) or (_phase >=0 and _phase < 36):
            c3_endo.append(1)
        else:
            c3_endo.append(0)

    c2_endo = []
    for _delta, _phase in zip(angle_d, angle_p):
        # C2 endo
        if (_delta >= 120 and _delta < 175) or (_phase >= 144 and _phase < 180):
            c2_endo.append(1)
        else:
            c2_endo.append(0)
        
    return c3_endo, c2_endo


def _intercalete(stacking_residue_index):    
    """
    Define intercaleted structures.
    
    RNA structures are intercalated if nucleotide `j` inserts between and stacks against nb `i` and `i+1`.
    
    Parameters
    ----------

    stacking_residue_index : list
        List of stacking residue index (e.g. [[0, 1], [1, 2]]).

    Returns
    -------

    name : str
        Category name
    """
    
    name = ""
    
    # 1-'0'-2-3
    # 2-'0'-1-3
    if [[0, 1], [0, 2]] == stacking_residue_index:
        name = "I0102"
            
    # 1-'2'-0-3
    # 0-'2'-1-3
    if [[0, 2], [1, 2]] == stacking_residue_index:
        name = "I0212"

    # 1-2-'0'-3
    if [[0, 2], [0, 3]] == stacking_residue_index:
        name = "I0203"

    # 0-'3'-1-2
    if [[0, 3], [1, 3]] == stacking_residue_index:
        name = "I0313"
            
    # 0-2-'1'-3
    # 0-3-'1'-2
    if [[1, 2], [1, 3]] == stacking_residue_index:
        name = "I1213"
        
    # 0-2-'3'-1
    # 0-1-'3'-2
    if [[1, 3], [2, 3]] == stacking_residue_index:
        name = "I1323"
        
    # 1-'2'-'0'-3
    if [[0, 2], [0, 3], [1, 2]] == stacking_residue_index:
        name = "I020312"        

    # 0-'2'-'1'-3
    if [[0, 2], [1, 2], [1, 3]] == stacking_residue_index:
        name = "I021213"

    # 0-'3'-'1'-2
    if [[0, 3], [1, 2], [1, 3]] == stacking_residue_index:
        name = "I031213"
    
    return name



def _tangled(stacking_residue_index):    
    """
    Define tangled structures. (>2 nb stacking with 5' and 3' stacking)            
    
    RNA structures are tangled if 5' and 3' nb are stacking and number of stacking nb are 3.
    
    Parameters
    ----------

    stacking_residue_index : list
        List of stacking residue index (e.g. [[0, 1], [1, 2]]).

    Returns
    -------

    name : str
        Category name
    """
    
    name = ""        

    # 1-2-'3'-0
    #if [[0, 3], [1, 2]] == stacking_residue_index:
    #    name = "T0312"

    #if [[0, 3], [2, 3]] == stacking_residue_index:
    #    name = "T0323"
    
    if [[0, 3], [1, 2], [2, 3]] == stacking_residue_index:
        name = "T031223"
    
    # 3-'0'-1-2
    #if [[0, 1], [0, 3]] == stacking_residue_index:
    #    name = "T0103"
    
    #if [[0, 1], [1, 2]] == stacking_residue_index:
    #    name = "T0103"
    
    if [[0, 1], [0, 3], [1, 2]] == stacking_residue_index:
        name = "T010312"
        
    return name



def annotate(replica_index, alpha, beta, gamma, delta, eps, zeta, chi, phase, stacking):
    """
    Annotate RNA structures.

    Cateogries RNA tetramer structure into 6 categories based on their geometries.

    AMa: A-form major
    AMi: A-form minor
    I:   Intercaleted
    F1:  First nb flipped
    F4:  Last nb flipped
    O:   Others

    Parameters
    -----------
    replica_index : int
        Replica index number.
    alpha : np.ndarray
        Backbone alpha angles in degrees of shape [n_iterations, n_residues, n_angles].
    beta : np.ndarray
        Backbone beta angles in degrees of shape [n_iterations, n_residues, n_angles].
    gamma : np.ndarray
        Backbone gamma angles in degrees of shape [n_iterations, n_residues, n_angles].
    delta : np.ndarray
        Backbone delta angles in degrees of shape [n_iterations, n_residues, n_angles].
    eps : np.ndarray
        Backbone epsilon angles in degrees of shape [n_iterations, n_residues, n_angles].
    zeta : np.ndarray
        Backbone zeta angles in degrees of shape [n_iterations, n_residues, n_angles].
    chi : np.ndarray
        Nucleobase chi angles in degrees of shape [n_iterations, n_residues, n_angles].
    phase : np.ndarray
        Sugar pucker angles in degrees of shape of shape [n_iterations, n_residues, n_angles].
    stacking : np.ndarray
        Stacking information shape of [n_iterations, stacking_pattern]. Stacking_pattern contains stacking residue index and stacking form (e.g. [[[0, 1], [1, 2], [2, 3]], ['>>', '>>', '>>']]).

    Return
    ----------
    xxxx
    
    """    
    obs_dict = defaultdict(list)
    unknown_category = defaultdict(list)
    myclass, myclass_by_number = [], []
    
    for frame_idx in range(len(stacking)):
        # stackings[frame_idx] : list
        #    e.g. [[[0, 1], [1, 2], [2, 3]], ['>>', '>>', '>>']]

        names = []
        stacking_residue_index = stacking[frame_idx][0]
        stacking_pattern = stacking[frame_idx][1]
        
        # A-form
        if stacking_residue_index == [[0, 1], [1, 2], [2, 3]]:
            c3_binary, c2_binary = _check_endo(delta[frame_idx], phase[frame_idx])
            if stacking_pattern == ['>>', '>>', '>>'] and sum(c3_binary) == 4:
                names.append("AMa")
            else:
                names.append("AMi")

        # Partial stacking
        if stacking_residue_index == [[1, 2], [2, 3]] and stacking_pattern == ['>>', '>>']:
            names.append("F1")
        if stacking_residue_index == [[0, 1], [1, 2]] and stacking_pattern == ['>>', '>>']:
            names.append("F4")

        # Intercalete
        if len(stacking_pattern) >= 2:
            _name = _intercalete(stacking_residue_index)
            if _name.startswith("I"):
                names.append("I")

        # Other
        if len(stacking_pattern) == 0 or len(stacking_pattern) == 1:
            names.append("O")
        if len(names) == 0:
            names.append("O")
            unknown_category[str(stacking_pattern) + str(stacking_residue_index)].append(frame_idx+1)

        assert len(names) == 1, "{}: multiple annotation {}\t{}\t{}".format(frame_idx+1, names, stacking_pattern, stacking_residue_index)
        
        for category in myclass_mapping_dict.keys():
            if names[0] == category:
                obs_dict[category].append(1)
            else:
                obs_dict[category].append(0)        
        
        myclass.append(names[0])
        myclass_by_number.append(myclass_mapping_dict[names[0]])

    return myclass, myclass_by_number, obs_dict, unknown_category



# ==============================================================================
# MAIN AND RUN
# ==============================================================================

def run(kwargs):
    """
    Reweight conformational populations using PyMBAR. Each REPX simulation are analyzed consecutively and effective energies are used to decorrelate the samples.
    All conformations are annotated as AMa, AMi, I, F1, F4, or O and its populations will be reweighted using mbar weights.

    Parameters
    ----------
    kwargs : dictionary
        Keyword options to run the analysis. 
    """
    npzfile = kwargs["npzfile"]
    ncfile = kwargs["ncfile"]
    output_prefix = kwargs["output_prefix"]
    plot_title = kwargs["plot_title"]


    if not os.path.exists(output_prefix) and output_prefix != ".":
        os.mkdir(output_prefix)
    if not os.path.exists(npzfile):
        raise Exception("npzfile not found")
    if not os.path.exists(ncfile):
        raise Exception("netcdf not found")
    if plot_title == "":
        plot_title = None


    # ----
    # Load data
    # ----
    print(">load data")

    # npz 
    npzfile = np.load(npzfile, allow_pickle=True)
    bb_angles = radian_to_degree(npzfile["bb_angles"])         # [n_replicas, n_iterations, n_residues, n_angles]
    pucker_angles = radian_to_degree(npzfile["pucker_angles"]) # [n_replicas, n_iterations, n_residues, n_angles]
    stackings = npzfile["stackings"]                           # [n_replicas, n_iterations, stacking_info]
    _, _, n_residues, n_angles = bb_angles.shape               # [n_replicas, n_iterations, n_residues, n_jcouplings]
    print("found {} observables for {} residues".format(n_angles, n_residues))
    
    # trajectory
    reporter = mmtools.multistate.MultiStateReporter(ncfile, open_mode='r')
    analyzer = mmtools.multistate.MultiStateSamplerAnalyzer(reporter)


    # -----
    # Annotate structures
    # -----
    print(">annotate structures")

    myclasses_by_number = []
    for replica_index in range(analyzer.n_replicas):
        alpha, beta, gamma, delta, eps, zeta, chi, phase, stacking = extract_data(replica_index, bb_angles, pucker_angles, stackings)
        
        # myclass: list of annotated category by names.
        # myclass_by_number: list of annotated category by integers.
        # obs_dict: defaultdict that stores all categories. Cateogry names are used as keys and stores lists of 1 (True) or 0 (False) if the structure is annotated to that category class.
        # unknown_category: defaultdict that was assigned to any of the given categories.
        myclass, myclass_by_number, obs_dict, unknown_cateorgy = annotate(replica_index, alpha, beta, gamma, delta, eps, zeta, chi, phase, stacking)

        # Store category numbers for each replica
        myclasses_by_number.append(myclass_by_number) 

        # Count each class (annotation category)
        from collections import Counter
        d = Counter(myclass)
        mydata = {
            "AMa": d["AMa"], \
            "AMi": d["AMi"], \
            "I":   d["I"], \
            "F1":  d["F1"], \
            "F4":  d["F4"], \
            "O":   d["O"]
        }
        
        print("replica {}:\t{}".format(replica_index, mydata))

    # Check shape
    myclasses_by_number = np.array(myclasses_by_number)
    assert analyzer.n_replicas == myclasses_by_number.shape[0]

    # Reformat array for decorrelation detection
    k, n = myclasses_by_number.shape
    o_kn = np.zeros([k, n+1])  # add iteration to match the shape size with energy matrix
    o_kn[:,1:] = myclasses_by_number


    # -----
    # Compute decorrelated energies and observables
    # https://github.com/choderalab/openmmtools/blob/1abb4f8112d231c2bbe1b42723ac692c50da631c/openmmtools/multistate/multistateanalyzer.py#L1468
    # -----
    print(">compute decorrelated energies and observables")

    # Energy_data is [energy_sampled, energy_unsampled, neighborhood, replicas_state_indices]
    energy_data = list(analyzer.read_energies())
    # Generate the equilibration data
    sampled_energy_matrix, unsampled_energy_matrix, neighborhoods, replicas_state_indices = energy_data
    # Note: This is different from pymbar.timeseries.detectEquilibration. analyzer._get_equilibration_data uses max_subset and excludes first iteration (minimization) to detect the equilibration data.
    number_equilibrated, g_t, Neff_max = analyzer._get_equilibration_data(sampled_energy_matrix, neighborhoods, replicas_state_indices)

    # Remove equilibrated and decorrelated data from energy_data
    for i, energies in enumerate(energy_data):
        # Discard equilibration iterations.
        energies = mmtools.multistate.utils.remove_unequilibrated_data(energies, number_equilibrated, -1)
        # Subsample along the decorrelation data.
        energy_data[i] = mmtools.multistate.utils.subsample_data_along_axis(energies, g_t, -1)
    sampled_energy_matrix, unsampled_energy_matrix, neighborhood, replicas_state_indices = energy_data

    # Initialize the MBAR matrices in ln [n_state, n_replica * n_iteration] form.
    n_replicas, n_sampled_states, n_iterations = sampled_energy_matrix.shape
    _, n_unsampled_states, _ = unsampled_energy_matrix.shape
    n_total_states = n_sampled_states + n_unsampled_states
    energy_matrix = np.zeros([n_total_states, n_iterations*n_replicas])
    samples_per_state = np.zeros([n_total_states], dtype=int)

    # Compute shift index for how many unsampled states there were.
    # This assume that we set an equal number of unsampled states at the end points.
    first_sampled_state = int(n_unsampled_states/2.0)
    last_sampled_state = n_total_states - first_sampled_state

    # Cast the sampled energy matrix from kln' to ln form.
    energy_matrix[first_sampled_state:last_sampled_state, :] = analyzer.reformat_energies_for_mbar(sampled_energy_matrix)
    # Determine how many samples and which states they were drawn from.
    unique_sampled_states, counts = np.unique(replicas_state_indices, return_counts=True)
    # Assign those counts to the correct range of states.
    samples_per_state[first_sampled_state:last_sampled_state][unique_sampled_states] = counts

    # Cast decorrelated energies and states
    decorrelated_u_ln = energy_matrix
    decorrelated_N_l = samples_per_state

    # Decorrelate observables
    import copy
    _o_kn = copy.deepcopy(o_kn)
    _o_kn = mmtools.multistate.utils.remove_unequilibrated_data(_o_kn, number_equilibrated, -1)
    decorrelated_o_kn = mmtools.multistate.utils.subsample_data_along_axis(_o_kn, g_t, -1)
    obs = decorrelated_o_kn.flatten()


    # -----
    # MBAR
    # -----
    print(">analyze with mbar")
    mbar = MBAR(decorrelated_u_ln, decorrelated_N_l)
    weights = mbar.getWeights()[:,0]   # weights for the first thermodynamic state

    mbar_results = {}
    for k, v in myclass_mapping_dict.items():
        indices = np.where(obs == v)
        population = weights[indices].sum() * 100
        count = 100*len(indices[0])/len(obs)
        mbar_results[k] = {"count": count, "reweight": population}

    
    # -----
    # SAVE RESULTS
    # -----
    print(">plot")
    population = [ v["reweight"] for v in mbar_results.values() ]
    logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)   # change logging level
    plot(x=mbar_results.keys(), y=population, yerr=None, plot_title=plot_title, output_prefix=output_prefix)

    print(">save summary")
    with open('{}/population.txt'.format(output_prefix), 'w') as f:
        f.write("NAME\tCOUNT\tPOPULATION\n")
        for k, v in mbar_results.items():
            f.write("{}\t{:.2f}\t{:.2f}\n".format(k, v["count"], v["reweight"]))

    print(">save decorrelated data")
    np.savez("mbar_population_decorrelated.npz", u_ln=decorrelated_u_ln, N_l=decorrelated_N_l, o_kn=decorrelated_o_kn)



@click.command()
@click.option("--npzfile", required=True, default="mydata.npz",    type=str, help="filename of pre-calculated data in relative path")
@click.option("--ncfile",  required=True, default="../enhanced.nc", type=str, help="filename of repx trajectory in relative path")
@click.option("--plot_title",    default="",  type=str, help="plot title")
@click.option("--output_prefix", default=".", type=str, help="output prefix to save output files")
def cli(**kwargs):
    run(kwargs)



if __name__ == '__main__':
    cli()
