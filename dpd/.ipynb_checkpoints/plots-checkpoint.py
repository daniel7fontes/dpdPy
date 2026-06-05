"""
================================================================
Utilities for spectrum and constellation plots (:mod:`dpd.plots`)
================================================================

   plotConst  -- Plot the transmitted and received symbols constellations.
   plotSpec   -- Plot the transmitted and received signal's power spectral density (PSD).
   
"""

"""Plot utilities."""


import numpy as np
import matplotlib.pyplot as plt


def plotConst(symbTx, symbRx, axs_lim = 1.5, show = True, save = False, file_path = None):
    """
    Plot the transmitted and received symbols constellations.
    
    Parameters
    ----------
    symbTx : np.array
        Complex-valued transmitted symbols
    
    symbRx : list or np.array
        Complex-valued received symbols
    
    axs_lim : float
        Max and min x/y limits. Default is 1.5

    show : bool
        A flag that indicates whether the plot is displayed or not. Default is True
        
    save : bool
        A flag that indicates whether the plot is saved or not. Default is False
    
    file_path : string 
        Path where the plot will be stored. Default is None.
    
    """
    
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
    """
    Plot the transmitted and received signal's power spectral density (PSD).
    
    Parameters
    ----------
    freq : np.array
        Frequency points corresponding to the PSD values
    
    P_sigTx : np.array
        PSD of the transmitted signal
    
    P_sigRx : list or np.array
        PSD of the received signals
    
    x_lim : list
        X-axis limits. Default is [-2, 2]
        
    y_lim : list
        Y-axis limits. Default is [-125, -80]

    freq_unit : string
        Frequency unit for frequency plot ("THz", "GHz", "MHz" or "KHz"). Default is "GHz"

    show : bool 
        A flag that indicates whether the plot is displayed or not. Default is True
        
    save : bool
        A flag that indicates whether the plot is saved or not. Default is False
    
    file_path : string 
        Path where the plot will be stored. Default is None.
    
    """
    
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

    axs.legend(framealpha = 1, fontsize = 14)
    
    plt.grid()    
    plt.tight_layout()
    
    if save:
        plt.savefig(file_path)
    
    if not(show):
        plt.close()