#!/usr/bin/env python
"""
Reweight RNA tetramer j3-coupling with PyMBAR
=============================================

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
import barnaba as bb
from barnaba import definitions
from barnaba.nucleic import Nucleic
import openmmtools as mmtools
from pymbar import MBAR


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

# Jcoupling names and index
# Couplings are pre-calculated in the order of  "H1H2", "H2H3", "H3H4", "1H5P", "2H5P", "1H5H4", "2H5H4", "H3P".
dict_coupling_mapping_by_index = {}
dict_coupling_mapping_by_index[0] = "nu1"  
dict_coupling_mapping_by_index[1] = "nu2"  
dict_coupling_mapping_by_index[2] = "nu3"  
dict_coupling_mapping_by_index[3] = "beta1"  
dict_coupling_mapping_by_index[4] = "beta2"  
dict_coupling_mapping_by_index[5] = "gamma1"  
dict_coupling_mapping_by_index[6] = "gamma2"  
dict_coupling_mapping_by_index[7] = "epsilon" 

# Not used but listed for memo
#dict_coupling_mapping_by_atom_name = {}
#dict_coupling_mapping_by_atom_name["H1H2"] = "nu1"  
#dict_coupling_mapping_by_atom_name["H2H3"] = "nu2"  
#dict_coupling_mapping_by_atom_name["H3H4"] = "nu3"  
#dict_coupling_mapping_by_atom_name["1H5P"] = "beta1"  
#dict_coupling_mapping_by_atom_name["2H5P"] = "beta2"  
#dict_coupling_mapping_by_atom_name["1H5H4"] = "gamma1"  
#dict_coupling_mapping_by_atom_name["2H5H4"] = "gamma2"  
#dict_coupling_mapping_by_atom_name["H3P"] = "epsilon" 



# ==============================================================================
# SUBROUTINE
# 
# Since `openmmtools.multistate.multistateanalyzer._get_equilibration_data` uses 
# effective energy to detect equilibration, manually compute equilibration 
# interation (n_equilibration), statistical insufficiency (g_t), and effective number of 
# uncorrelated samples (n_effective_max) following similar procedures from 
# https://github.com/choderalab/openmmtools/blob/1abb4f8112d231c2bbe1b42723ac692c50da631c/openmmtools/multistate/multistateanalyzer.py#L2009
# ==============================================================================

def load_benchmark_data(benchmark_path, seq, keyname):
    """
    """
    import yaml
    yfile = os.path.join(benchmark_path, seq, "00_data", "experiment.yml")
    with open(yfile, "r") as file:
        d = yaml.safe_load(file)
    
    param_names = []
    for res in d['experiment_1']['measurement'].keys():
        names = d['experiment_1']['measurement'][res].keys()
        param_names = [ _name.replace('_', '') for _name in names ]
        break

    try:
        mydict = {}
        for res in d[keyname]['measurement'].keys():
            names = d[keyname]['measurement'][res].keys()
            
            vals = []
            for name in names:
                v = d[keyname]['measurement'][res][name]['value']
                #print(res, name, v)
                if v == None:
                    v = 0
                vals.append(v)
            mydict[res] = vals

        # Convert to pandas
        df = pd.DataFrame.from_dict(mydict)
        df.index = param_names
        df = df.T
    except:
        pass

    return df


def plot(result_dict_per_residue, output_prefix, benchmark_path, seq):
    """
    """
    df_exp = load_benchmark_data(benchmark_path, seq, keyname="experiment_1")
    df_rev = load_benchmark_data(benchmark_path, seq, keyname="computational_1") 
    try:
        # DE Shaw Amber14ff (might not be present for some tetramers)
        df_a14 = load_benchmark_data(benchmark_path, seq, keyname="computational_2") 
    except:
        import copy
        df_a14 = copy.deepcopy(df_exp)
        df_a14.loc[:,:] = 0

    pd.options.display.float_format = '{:.2f}'.format
    df = pd.DataFrame(result_dict_per_residue).T

    # Plot
    import matplotlib.colors as mcolors
    mycolors = mcolors.TABLEAU_COLORS

    names = ["beta1", "beta2", "gamma1", "gamma2", "epsilon", "nu1", "nu2", "nu3"]
    fig, axes = plt.subplots(4,1, figsize=(18, 12), sharex=True)
    for i, resname in enumerate(result_dict_per_residue.keys()):
        xpos = 0
        for name, mycolor in zip(names, mycolors):
            # Hide values with zero
            exp_scale, rev_scale, a14_scale, scale = 1, 1, 1, 1
            if df_exp[name][i] == 0:
                exp_scale = 0
            if df_rev[name][i] == 0:
                rev_scale = 0
                scale = 0
            if df_a14[name][i] == 0:
                a14_scale = 0

            axes[i].scatter(xpos-0.1, df_exp[name][i], marker='x', s=60 * exp_scale, c=mycolors[mycolor])
            axes[i].scatter(xpos, df_rev[name][i], marker='^', s=60 * rev_scale, c=mycolors[mycolor])
            if df_a14.sum().min() != 0:
                axes[i].scatter(xpos+0.1, df_a14[name][i], marker='_', s=60 * a14_scale, c=mycolors[mycolor])
                try:
                    axes[i].errorbar(xpos+0.2, df[name][i]['mu'], yerr=df[name][i]['sigma'], fmt='o', capsize=6 * scale, markersize=10 * scale, c=mycolors[mycolor])
                except:
                    pass
            else:
                try:
                    axes[i].errorbar(xpos+0.1, df[name][i]['mu'], yerr=df[name][i]['sigma'], fmt='o', capsize=6 * scale, markersize=10 * scale, c=mycolors[mycolor])
                except:
                    pass

            # Axes
            axes[i].set_title(resname, x=0.03, y=0.75, fontsize=24)
            axes[i].yaxis.set_minor_locator(AutoMinorLocator())
            axes[i].yaxis.set_ticks_position("left")
            axes[i].set_ylim(-1.5,13.5)
            axes[i].set_xlim(-0.5,7.5)

            # Increment position
            xpos += 1

    axes[i].set_xticks([0,1,2,3,4,5,6,7], [r"$\beta$1",r"$\beta$2",r"$\gamma$1",r"$\gamma$2",r"$\epsilon$",r"$\nu$1",r"$\nu$2",r"$\nu$3"])    
    axes[i].set_ylabel("$^3$J-coupling (Hz)")
    axes[i].yaxis.set_label_coords(-0.1, 2)
    axes[i].scatter(xpos, xpos, marker='x', s=60, c="k", label="Experimental")
    axes[i].scatter(xpos, xpos, marker='^', s=60, c="k", label="DEShaw.Revised")
    if df_a14.sum().min() != 0:
        axes[i].scatter(xpos, xpos, marker='_', s=60, c="k", label="DEShaw.Amber.OL3")
    axes[i].scatter(xpos, xpos, marker='o', s=60, c="k", label="HREMD.Amber.OL3")
    axes[i].legend(bbox_to_anchor=(0.23, 4.05), fontsize=16)

    plt.subplots_adjust(hspace=0.1, wspace=0.1)
    plt.tight_layout()
    plt.savefig("{}/coupling.png".format(output_prefix))
    plt.close()
    
    df.to_pickle("{}/coupling.pkl".format(output_prefix))



def get_effective_observable_timeseries(u_kn):
    """
    """
    n_replicas, n_frames = u_kn.shape
    u_n = np.zeros([n_frames], np.float64)
    
    # Slice of all replicas, have to use this as : is too greedy
    replicas_slice = range(n_replicas)
    for iteration in range(n_frames):
        # Slice the current sampled states by those replicas.
        u_n[iteration] = np.sum(u_kn[replicas_slice, iteration])

    return u_n


# Not a good way to pass analyzer?
def _get_equilibration_data_custom(u_kn, analyzer):
    """
    """
    # Compute effective observable timeseries
    u_n = get_effective_observable_timeseries(u_kn)
    
    # For SAMS, if there is a second-stage start time, use only the asymptotically optimal data
    # if self._n_equilibration_iterations was not specified, discard minimization frame
    t0 = analyzer._n_equilibration_iterations if analyzer._n_equilibration_iterations is not None else 1

    # Discard equilibration samples
    max_subset = 100   # default setting in openmmtools
    i_t, g_i, n_effective_i = mmtools.multistate.utils.get_equilibration_data_per_sample(u_n[t0:], max_subset=max_subset)
    n_effective_max = n_effective_i.max()
    i_max = n_effective_i.argmax()
    n_equilibration = i_t[i_max] + t0 
    g_t = analyzer._statistical_inefficiency if analyzer._statistical_inefficiency is not None else g_i[i_max]
    
    # Store equilibration data
    logger.debug(' number of iterations discarded to equilibration : {}'.format(n_equilibration))
    logger.debug(' statistical inefficiency of production region   : {}'.format(g_t))
    logger.debug(' effective number of uncorrelated samples        : {}'.format(n_effective_max))
    
    return n_equilibration, g_t, n_effective_max



# ==============================================================================
# MAIN AND RUN
# ==============================================================================

def run(kwargs):
    """
    Compute J-coupling for individual REPX trial.

    Parameters
    ----------

    Returns
    -------
    """
    npzfile = kwargs["npzfile"]
    ncfile = kwargs["ncfile"]
    seq = kwargs["seq"]
    benchmark_path = kwargs["benchmark_path"]
    output_prefix = kwargs["output_prefix"]
    start_frame = kwargs['start_frame']
    end_frame = kwargs['end_frame']
    skip_frame = kwargs['skip_frame']

    if not os.path.exists(output_prefix) and output_prefix != ".":
        os.mkdir(output_prefix)
    if not os.path.exists(npzfile):
        raise Exception("npzfile not found")
    if not os.path.exists(ncfile):
        raise Exception("netcdf not found")

    # Define residue names
    resnames = [ s.upper() + str(i+1) for i, s in enumerate(seq) ]
    print(">residue names: {}".format(resnames))
    

    #
    # Load data
    #
    print(">load data")

    # npz 
    npzfile = np.load(npzfile, allow_pickle=True)
    couplings = npzfile['couplings']
    _, _, n_residues, n_couplings = couplings.shape  # [n_replicas, n_frames, n_residues, n_jcouplings]
    print(">observable shape is {}".format(couplings.shape))
    print(">found {} observables for {} residues".format(n_couplings, n_residues))
    
    # trajectory
    reporter = mmtools.multistate.MultiStateReporter(ncfile, open_mode='r')
    analyzer = mmtools.multistate.MultiStateSamplerAnalyzer(reporter)
    n_iterations, n_replicas, n_states = analyzer.n_iterations, analyzer.n_replicas, analyzer.n_states
    print(">found {} iterations, {} replicas, and {} states".format(n_iterations, n_replicas, n_states))


    #
    # Preprocess observable data to perform decorrelation analysis
    #
    print(">reformat observable")
    dict_per_residue = {}  # key: residue name
    for residue_index in range(n_residues):
        dict_per_coupling = {}  # key: jcoupling name
        for coupling_index in range(n_couplings):
            resname = resnames[residue_index]
            o_kn = couplings[:,:,residue_index,coupling_index]  # [n_replicas, n_frames, n_residues, n_observables]
            dict_per_coupling[dict_coupling_mapping_by_index[coupling_index]] = o_kn
        dict_per_residue[resnames[residue_index]] = dict_per_coupling


    #
    # Compute equilibration iteration and statistical inefficiency (decorrelated iteration steps) for each J-couplings for every residue
    #
    print(">detect equilibration")
    dict_equilibration_data = {}
    for residue_key in dict_per_residue.keys():    
        tmp_dict = {}
        for coupling_key in dict_per_residue[residue_key].keys():
            logger.debug("residue: {} ({})".format(residue_key, coupling_key)) 
            # Compute equilibration
            u_kn = dict_per_residue[residue_key][coupling_key]
            n_equilibration, g_t, n_effective_max = _get_equilibration_data_custom(u_kn, analyzer)
            # Store equilibration data
            tmp_dict[coupling_key] = [n_equilibration, g_t, n_effective_max]
        dict_equilibration_data[residue_key] = tmp_dict


    #
    # Load energy data from analyzer.read_energies() 
    #
    energy_data_tmp = list(analyzer.read_energies())
    # Generate the equilibration data
    sampled_energy_matrix, unsampled_energy_matrix, neighborhoods, replicas_state_indices = energy_data_tmp
    # Slice array
    if end_frame < 0:
        end_frame = sampled_energy_matrix.shape[-1] + end_frame + 1
    sampled_energy_matrix = sampled_energy_matrix[:,:,start_frame:end_frame:skip_frame]
    unsampled_energy_matrix = unsampled_energy_matrix[:,:,start_frame:end_frame:skip_frame]
    neighborhoods = neighborhoods[:,:,start_frame:end_frame:skip_frame]
    replicas_state_indices = replicas_state_indices[:,start_frame:end_frame:skip_frame]
    # Update energy_data
    energy_data = [sampled_energy_matrix, unsampled_energy_matrix, neighborhoods, replicas_state_indices]
    # Check array shape
    assert o_kn.shape == replicas_state_indices.shape


    #
    # Compute decorrelated energies and observables
    #
    print(">compute decorrelated energies and observables")
    dict_decoupled_data = {}  # key: residue name
    for residue_key in dict_equilibration_data.keys():
        tmp_dict = {} # key: coupling name
        for coupling_key in dict_equilibration_data[residue_key]:
            # Get equilibrated iterations to discard and statistical inefficiency (decorrelated iteration steps)
            number_equilibrated, g_t, Neff_max = dict_equilibration_data[residue_key][coupling_key]
            logger.debug(">residue: {} ({}) number_equilibrated: {}, g_t: {}, Neff_max: {}".format(residue_key, coupling_key, number_equilibrated, g_t, Neff_max))
            
            if np.isnan(g_t) or np.isnan(Neff_max):
                # Some couplings could be None for C5'- and C3'- nucleotides
                pass
            else:

                #
                # Energy
                #
                import copy
                _energy_data = copy.deepcopy(energy_data)
                for i, energies in enumerate(_energy_data):
                    # Discard equilibration iterations.
                    energies = mmtools.multistate.utils.remove_unequilibrated_data(energies, number_equilibrated, -1)
                    # Subsample along the decorrelation data.
                    _energy_data[i] = mmtools.multistate.utils.subsample_data_along_axis(energies, g_t, -1)    
                sampled_energy_matrix, unsampled_energy_matrix, neighborhood, replicas_state_indices = _energy_data

                # Initialize the MBAR matrices in ln form.
                n_replicas, n_sampled_states, n_frames = sampled_energy_matrix.shape
                _, n_unsampled_states, _ = unsampled_energy_matrix.shape

                # We assume there are no unsampled states. Altough below is redundant, we will follow similar procedures found in 
                # openmmtools.multistate.multistateanalyzer._compute_mbar_decorrelated_energies
                assert n_unsampled_states == 0
                n_total_states = n_sampled_states
                energy_matrix = np.zeros([n_total_states, n_frames*n_replicas])
                samples_per_state = np.zeros([n_total_states], dtype=int)
                # Compute shift index for how many unsampled states there were.
                # This assumes that we set an equal number of unsampled states at the end points.
                first_sampled_state = 0
                last_sampled_state = n_total_states
                # Cast the sampled energy matrix from kln' to ln form.
                energy_matrix[first_sampled_state:last_sampled_state, :] = analyzer.reformat_energies_for_mbar(sampled_energy_matrix)
                # Determine how many samples and which states they were drawn from.
                unique_sampled_states, counts = np.unique(replicas_state_indices, return_counts=True)
                # Assign those counts to the correct range of states.
                samples_per_state[first_sampled_state:last_sampled_state][unique_sampled_states] = counts

                # Cast decorrelated energies and state
                decorrelated_u_ln = energy_matrix
                decorrelated_N_l = samples_per_state

                #
                # Observables
                #
                # Remove equilibrated and decorrelated data from observable
                o_kn = dict_per_residue[residue_key][coupling_key]
                _o_kn = copy.deepcopy(o_kn)
                # Discard equilibration iterations.
                _o_kn = mmtools.multistate.utils.remove_unequilibrated_data(_o_kn, number_equilibrated, -1)
                # Subsample along the decorrelation data.
                decorrelated_o_kn = mmtools.multistate.utils.subsample_data_along_axis(_o_kn, g_t, -1)
                # Reformat
                decorrelated_o_n = decorrelated_o_kn.flatten()
            
                # Temporarily store (tmp_dict will be overwritten)
                tmp_dict[coupling_key] = {'u_ln': decorrelated_u_ln, 'N_l': decorrelated_N_l, 'o_n': decorrelated_o_n}
        
        # Store
        dict_decoupled_data[residue_key] = tmp_dict


    #
    # MBAR
    #
    print(">analyze with mbar")

    dict_reweighted_coupling = {}  # key: residue
    for residue_key in dict_decoupled_data.keys():
        tmp_dict = {}
        for coupling_key in dict_decoupled_data[residue_key].keys():
            #print(residue_key, coupling_key)
            u_ln = dict_decoupled_data[residue_key][coupling_key]['u_ln']
            N_l = dict_decoupled_data[residue_key][coupling_key]['N_l']
            o_n = dict_decoupled_data[residue_key][coupling_key]['o_n']
            assert u_ln.shape[0] == N_l.size and u_ln.shape[1] == o_n.size
            
            # mbar
            mbar = MBAR(u_ln, N_l)      
            u0 = u_ln[0,:]   # energy from state 0
            results = mbar.computeExpectations(o_n, u0, return_dict=True)

            #mu, sigma = float(results['mu']), float(results['sigma'])
            tmp_dict[coupling_key] = results
            
        dict_reweighted_coupling[residue_key] = tmp_dict



    # 
    # Save (decorrelated data and plots)
    # 
    print(">plot")
    plot(dict_reweighted_coupling, output_prefix, benchmark_path, seq)

    print(">save summary")
    with open('{}/coupling.txt'.format(output_prefix), 'w') as f:
        f.write("RESNAME\tJCOUPLING\tAVERAGE\tSIGMA\n")
        for resname in dict_reweighted_coupling.keys():
            for k, v in dict_reweighted_coupling[resname].items():
                #print(mdtrial, resname, k, v)
                try:
                    f.write("{}\t{}\t{:.2f}\t{:.2f}\n".format(resname, k, float(v['mu']), float(v['sigma'])))
                except:
                    pass

    print(">save decorrelated data")
    np.savez("mbar_coupling_decorrelated.npz", mydict=dict_equilibration_data)



@click.command()
@click.option("--npzfile", required=True, default="mydata.npz",    type=str, help="filename of pre-calculated data in relative path")
@click.option("--ncfile",  required=True, default="../enhanced.nc", type=str, help="filename of repx trajectory in relative path")
@click.option("--seq", required=True, type=click.Choice(["aaaa", "cccc", "uuuu", "gacc", "caau"]), help="tetramer sequence")
@click.option("--benchmark_path", required=True, type=str, help="path to benchmark directory")
@click.option("--output_prefix", default=".", type=str, help="output prefix to save output files")
@click.option("--start_frame", default=0, help="Index of the first frame to include in the trajectory. Index 0 corresponds to the minimization step.")
@click.option("--end_frame", default=-1, help="Index of the last frame to include in the trajectory")
@click.option("--skip_frame", default=1, help="Extract every n frames from the trajectory")
def cli(**kwargs):
    run(kwargs)



if __name__ == '__main__':
    cli()
