#!/usr/bin/env python

"""


AF 15/05/2022


"""

import matplotlib.pyplot as plt
import numpy as np
import reciprocalspaceship as rs
import scipy.optimize as opt
import gemmi as gm
import pandas as pd
import os
import mdtraj as md
from thor.scatter import atomic_formfactor
from scipy.stats import binned_statistic
import seaborn as sns
sns.set_context("notebook", font_scale=1.4)
from xtal_analysis.xtal_analysis_functions import load_mtz, res_cutoff


def get_formfactors(pdb, q_mag):
    
    """
    Given a pdb loaded in mdtraj: evaluates the form factor sum (sum((f_j)^2))
    for every atom j in the file at the stated q-vector magnitude (q_mag, float)
    """
    atomic_numbers = np.array([ a.element.atomic_number for a in pdb.topology.atoms ])
    form_factors   = []
    
    for i in atomic_numbers :
        form_factor   = atomic_formfactor(i, q_mag)
        form_factors.append(form_factor**2)
        
    return sum(form_factors)
    
    
def iso_wilson_fit(Is, ds, pdb, fit_cut):

    qs      = 2*np.pi/ds
    sum_fis = get_formfactors(pdb, qs)
    y_terms = Is/sum_fis
    x_terms = 1/(2*ds)**2
    
    x_terms = x_terms[np.isfinite(y_terms)]
    y_terms = y_terms[np.isfinite(y_terms)]
    
    def func(x, B, K): return np.exp(-2*B*x+K)
    params = opt.curve_fit(func,x_terms[np.where(x_terms >= fit_cut)], y_terms[np.where(x_terms >= fit_cut)])[0]

    return params[0], params[1], x_terms, y_terms
    
    

def iso_wilson_plot(x_terms, y_terms, bins_n, B, K, fit_cut):

    bins_x = np.linspace(np.min(x_terms)-1e-6, np.max(x_terms), bins_n)
    assignments = np.digitize(x_terms, bins_x, right=True)
    x_avg = np.bincount(assignments, weights=x_terms)[1:]/np.histogram(x_terms, bins=bins_x)[0]
    y_avg = np.bincount(assignments, weights=y_terms)[1:]/np.histogram(x_terms, bins=bins_x)[0]
    res_cut     = np.round(1/(2*np.sqrt(fit_cut)), decimals=2)


    fig, ax = plt.subplots(2,2, constrained_layout=True, figsize=(10,8))

    ax[0,0].scatter(x_terms, y_terms, color='lightblue', alpha=0.6, s=46, edgecolor='blue')
    ax[0,0].plot(x_avg, np.exp(-2*B*x_avg+K), 'r')
    ax[0,0].set_xlabel(r'sin$^2{\theta}$/$\lambda^2$')
    ax[0,0].set_ylabel(r'<Iobs>/$\Sigma f_j^2$')
    
    ax[0,1].scatter(x_terms, np.log(y_terms), color='lightblue', alpha=0.6, s=46, edgecolor='blue')
    ax[0,1].plot(x_avg, -2*B*x_avg+K, 'r')
    ax[0,1].set_xlabel(r'sin$^2{\theta}$/$\lambda^2$')
    ax[0,1].set_ylabel(r'ln(<Iobs>/$\Sigma f_j^2$)')
    
    ax[1,0].scatter(x_avg, y_avg, color='lightblue', alpha=0.6, s=46, edgecolor='blue')
    ax[1,0].plot(x_avg, np.exp(-2*B*x_avg+K), 'r')
    ax[1,0].set_xlabel(r'sin$^2{\theta}$/$\lambda^2$ binned')
    ax[1,0].set_ylabel(r'<Iobs>/$\Sigma f_j^2$ binned')
    
    ax[1,1].scatter(x_avg, np.log(y_avg), color='lightblue', alpha=0.6, s=46, edgecolor='blue')
    ax[1,1].plot(x_avg, -2*B*x_avg+K, 'r')
    ax[1,1].set_xlabel(r'sin$^2{\theta}$/$\lambda^2$ binned')
    ax[1,1].set_ylabel(r'ln(<Iobs>/$\Sigma f_j^2$) binned')

    fig.suptitle(r"Fitting Resolutions from {cut} $\AA$".format(cut=res_cut)+"\nB = {B} k = {k}".format(B=np.round(B, decimals=2), k=np.round(K, decimals=2)), fontsize=14)
    plt.show()
    fig.savefig("wilson-fits.png")


if __name__ == "__main__":

    fit_cut      = 0.06
    bins_n       = 20
    path         = "/Users/alisia/Desktop/2019-2022/JvT_group/kiiro-sfs/"
    
    off          = load_mtz("{}Alisia-deltaF-LCLS/PP-AllRuns-mosflm_fs-400nm-min50fs-1ps_truncate1.mtz".format(path)).dropna()
    dark_pdb     = md.load('LCLS-dark-coordinates.pdb')

    B, K, x_terms, y_terms = iso_wilson_fit(np.array(off['IMEAN_400nm']), np.array(off['dHKL']), dark_pdb, fit_cut)
    x_avg, y_avg           = iso_wilson_plot(x_terms, y_terms, bins_n, B, K, fit_cut)

