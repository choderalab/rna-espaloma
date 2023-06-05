#!/usr/bin/env python
# coding: utf-8

"""
Analyze equilibrated simulation.

Quick analysis to check pucker angles and rmsd distribution.
"""

import os, sys, math
import numpy as np
import glob
import mdtraj
import logging
import netCDF4 as nc
import warnings
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, FormatStrFormatter, AutoMinorLocator
import seaborn as sns
import barnaba as bb
from barnaba import definitions
from barnaba.nucleic import Nucleic


# logging
logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore")

# plot settings
pd.options.display.max_rows = None
pd.options.display.max_columns = None
pd.options.display.precision = 1
pd.options.display.float_format = '{:.1f}'.format

params = {'legend.fontsize': 40, 
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

plt.rcParams.update(params)


#
# BACKBONE DEFINITION
#
backbone_sugar_atoms = [
    "C1'", \
    "H1'", \
    "C2'", \
    "H2'", \
    "C3'", \
    "H3'", \
    "C4'", \
    "H4'", \
    "C5'", \
    "H5'", \
    "H5''", \
    "O2'", \
    "HO2'", \
    "O3'", \
    "O4'", \
    "O5'", \
    "P", \
    "OP1", \
    "OP2", \
    "HO5'", \
    "HO3'"
]



def radian_to_degree(a):
    """
    a : list
        [trajectory frame : residue : torsion]
    """
    
    a[np.where(a<0.0)] += 2.*np.pi
    a *= 180.0/np.pi

    # same as above
    #a = a*(180./np.pi)
    #a[np.where(a<0.0)] += 360
    
    return a



def calc_sugar_pucker(init_pdb, traj, rnames):
    """
    sugar pucker
    """
    # initial structure
    init_angles, res = bb.pucker_angles(init_pdb, topology=init_pdb)

    # equilibration
    angles, res = bb.pucker_rao_traj(traj)
    fig = plt.figure(figsize=(24,18))
    for i in range(len(rnames)):
        ax = fig.add_subplot(1, 4, i+1, polar=True)
        #ax.plot(polar=True)

        ax.scatter(angles[:,i,0], angles[:,i,1], s=10, c=np.arange(len(angles)), cmap='Blues', label="{}-{}".format(rnames[i], i))
        ax.scatter(init_angles[:,i,0], init_angles[:,i,1], marker="X", c="orange", edgecolors="black", s=150, linewidths=0.5)
        
        p3 = np.pi/5
        ax.text(0.5*p3, 1.6, "C3'-endo", ha='center', fontsize=16, fontweight='bold')
        ax.text(1.3*p3, 1.5, "C4'-exo",  ha='center', fontsize=16)
        ax.text(2.5*p3, 1.5, "O4'-endo", ha='center', fontsize=16)
        ax.text(3.7*p3, 1.5, "C1'-exo",  ha='center', fontsize=16)
        ax.text(4.5*p3, 1.6, "C2'-endo", ha='center', fontsize=16, fontweight='bold')
        ax.text(5.5*p3, 1.5, "C3'-exo",  ha='center', fontsize=16)
        ax.text(6.5*p3, 1.5, "C4'-endo", ha='center', fontsize=16)
        ax.text(7.5*p3, 1.6, "O4'-exo",  ha='center', fontsize=16)
        ax.text(8.5*p3, 1.5, "C1'-endo", ha='center', fontsize=16)
        ax.text(9.5*p3, 1.5, "C2'-exo",  ha='center', fontsize=16)
        
        xt = np.arange(0, 2*np.pi, p3)
        ax.set_xticks(xt)
        ax.set_yticks([])
        ax.set_ylim(0, 1.2)
        ax.tick_params(axis='both', labelsize=12)
        
        plt.tight_layout()
        #plt.legend(loc="upper center")
        plt.savefig("pucker_angles.png")



def calc_rmsd(init_traj, traj, rnames):
    """
    RMSD and eRMSD
    ---------

    RMSD:  Calculate rmsd after optimal alignment between reference and target structures. Superposition and RMSD calculations are performed using all heavy atoms. If the sequence of reference and target is different, only backbone/sugar heavy atoms are used.  
    eRMSD: Calculate ermsd between reference and target structures  
    """

    #
    # RMSD time plot
    #
    #init_traj = mdtraj.load(init_pdb)    
    rmsd = list(bb.functions.rmsd_traj(init_traj, traj))   
    rmsd = np.array(rmsd) * UNIT_NM_TO_ANGSTROMS
    
    #x = np.arange(1, len(rmsd)+1) * LOGGING_FREQUENCY * STRIDE * UNIT_PS_TO_NS
    x = np.arange(1, len(rmsd)+1) * LOGGING_FREQUENCY * UNIT_PS_TO_NS
    
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.set_title(PLOT_TITLE)

    # x-axis
    ax.set_xlabel(r'Time (ns)')
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    #ax.set_xlim([0, len(x)]) 
    
    # y-axis
    ax.set_ylabel(r'RMSD (${\rm \AA}$)')
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    ax.set_ylim([0, 6])

    ax.plot(x, rmsd, lw=1)
    plt.tight_layout()
    plt.savefig("rmsd.png")


    #
    # eRMSD time plot
    #
    ermsd = list(bb.functions.ermsd_traj(init_traj, traj))   
    ermsd = np.array(ermsd) * UNIT_NM_TO_ANGSTROMS

    fig, ax = plt.subplots(figsize=(12, 8))
    ax.set_title(PLOT_TITLE)

    # x-axis
    ax.set_xlabel(r'Time (ns)')
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    #ax.set_xlim([0, len(x)]) 

    # y-axis
    ax.set_ylabel(r'eRMSD (${\rm \AA}$)')
    ax.yaxis.set_minor_locator(AutoMinorLocator())

    ax.plot(x, ermsd, lw=1)
    plt.tight_layout()
    plt.savefig("ermsd.png")


    #
    # RMSD and eRMSD side-by-side plot
    #
    fig, (ax1, ax2) = plt.subplots(1,2, figsize=(24, 8))
    fig.suptitle(PLOT_TITLE, y=0.85)

    # xy-axis (1)
    ax1.set_xlabel(r'Time (ns)')
    ax1.xaxis.set_minor_locator(AutoMinorLocator())
    #ax1.set_xlim([0, len(x)]) 
    ax1.set_ylabel(r'RMSD (${\rm \AA}$)')
    ax1.yaxis.set_minor_locator(AutoMinorLocator())
    ax1.set_ylim([0, 6])

    # xy-axis (2)
    ax2.set_xlabel(r'Time (ns)')
    ax2.xaxis.set_minor_locator(AutoMinorLocator())
    #ax2.set_xlim([0, len(x)]) 
    ax2.set_ylabel(r'eRMSD (${\rm \AA}$)')
    ax2.yaxis.set_minor_locator(AutoMinorLocator())
    
    # plot
    ax1.plot(x, rmsd, lw=1, c='b')
    ax2.plot(x, ermsd, lw=1, c='r')

    plt.tight_layout()
    plt.savefig("rmsd_ermsd.png")




if __name__ == '__main__':
    name = sys.argv[1]
    input_prefix = sys.argv[2]

    PLOT_TITLE = "{} Espaloma".format(name.upper())
    UNIT_NM_TO_ANGSTROMS = 10
    UNIT_PS_TO_NS = 1/1000
    LOGGING_FREQUENCY = 100   # Default: 100 (unit: ps)

    # initial structure
    init_pdb = "{}/espaloma_mapped_solvated.pdb".format(input_prefix)
    _init_traj = mdtraj.load(init_pdb)
    # slice
    atom_indices = _init_traj.topology.select('not (protein or water or symbol Na or symbol Cl)')
    init_traj = _init_traj.atom_slice(atom_indices)

    # equilibrated
    ncfile = "../traj.nc"
    traj = mdtraj.load(ncfile, top=init_traj.topology)
    rnames = [ residue.name for residue in traj.topology.residues if residue.name not in ["HOH", "NA", "CL"] ]

    # calculate and plot
    calc_sugar_pucker(init_pdb, traj, rnames)
    calc_rmsd(init_traj, traj, rnames)

