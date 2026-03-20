# -*- coding: utf-8 -*-
"""
======================================================================
Funções para geração de figuras associadas à análise de resultados de DPD
======================================================================
"""

import numpy as np
import matplotlib.pyplot as plt
from optic.comm.metrics import fastBERcalc, calcEVM

def plotConst(symbTx, symbRx, axs_lim = 1.5, show = True, save = False, file_path = None):
    fig, axs = plt.subplots(figsize = (7, 7))
    
    if type(symbRx) == list:
        for symb in symbRx:
            axs.plot(symb.real, symb.imag, "o", ms = 1)

    else:
        axs.plot(symbRx.real, symbRx.imag, "o", ms = 1)
    
    axs.plot(symbTx.real, symbTx.imag, "o", color = "k",  ms = 3)

    axs.set_ylabel("Quadrature - Q")
    axs.set_xlabel("In-Phase - I")
    plt.axis("square")
    
    axs.set_xlim(-axs_lim, axs_lim)
    axs.set_ylim(-axs_lim, axs_lim)
    axs.minorticks_on()
    axs.tick_params(axis = 'both', top = "True", right = "True", which='minor',  width=1, direction = "in")
    axs.tick_params(axis = 'both', top = "True", right = "True", which='major',  width=1.5, direction = "in")

    plt.grid()
    
    plt.tight_layout()
    if save:
        plt.savefig(file_path)
    
    if not(show):
        plt.close()


def plotSpec(freq, P_sigTx, P_sigRx, label, x_lim = [-2, 2], y_lim = [-125, -80], freq_unit = "GHz", show = True, save = False, file_path = None):
    if freq_unit == "THz":
        freq_norm = 1e12
    
    elif freq_unit == "GHz":
        freq_norm = 1e9
    
    elif freq_unit == "MHz":
        freq_norm = 1e6
        
    elif freq_unit == "KHz":
        freq_norm = 1e3
    
    else:
        freq_norm = 1

    fig, axs = plt.subplots(1, 1, figsize = (7, 5))
    axs.plot(freq/freq_norm, 10*np.log10(P_sigTx), lw = 2)
    
    if type(P_sigRx) == list:
        for i, P_sig in enumerate(P_sigRx):
            axs.plot(freq/freq_norm, 10*np.log10(P_sig), lw = 2, label = label[i])

    else:
        axs.plot(freq/freq_norm, 10*np.log10(P_sigRx), lw = 2, label = label)
    
    
    axs.set_xlim(x_lim[0], x_lim[1])
    axs.set_ylim(y_lim[0], y_lim[1])
    
    axs.set_ylabel("Power Spectral Density [dB/Hz]")
    axs.set_xlabel(f"Frequency [{freq_unit}]")
    
    axs.minorticks_on()
    axs.tick_params(axis = 'both', top = "True", right = "True", which='minor',  width=1, direction = "in")
    axs.tick_params(axis = 'both', top = "True", right = "True", which='major',  width=1.5, direction = "in")

    axs.legend(framealpha = 1)
    
    plt.grid()    
    plt.tight_layout()
    
    if save:
        plt.savefig(file_path)
    
    if not(show):
        plt.close()
