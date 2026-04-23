"""
================================================================
Functions for metrics calculation (:mod:`dpd.calc_metrics`)
================================================================

   calcPAPR  -- Calculate the PAPR of a signal.
   calcACLR  -- Calculate the ACLR of a signal.
   calcMSE   -- Calculate the MSE between two signals.
   
"""

"""Utilities for PAPR, ACLR and MSE calculation."""


import numpy as np


def calcPAPR(x):
    """
    Calculate the Peak-to-Average Power Ratio (PAPR), in dB, of a signal x

    Parameters
    ----------
    x : np.array
        Input signal
    
    Returns
    -------
    PAPR : float
        Peak-to-Average Power Ratio of x (in dB)
        
    """
    peak_power    = np.max(np.abs(x)) ** 2
    average_power = np.mean(np.abs(x) ** 2)
    
    PAPR = 10 * np.log10(peak_power / average_power)
    
    return PAPR


def calcACLR(Psd, freqs, B, offset):
    """
    Calculate the Adjacent Channel Leakage Ratio (ACLR) of a signal with given power spectral density (Psd)

    Parameters
    ----------
    Psd : np.array
       Power spectral density of the signal 
    
    freqs : np.array
       Frequency points corresponding to the PSD values 
    
    B : float
        Frequency limit of the in-band signal frequency range
    
    offset : 
        Frequency offset for the in-band signal frequency range limit
        
    Returns
    -------
    ACLR : float
        Adjacent channel leakage ratio of the signal with power spectral density Psd
        
    """
    df = freqs[1] - freqs[0]
    
    Pin = np.sum(Psd[freqs >= - B] * df) - np.sum(Psd[freqs >= B] * df)
    
    Pout1 = np.sum(Psd[ freqs <= -B - offset] * df) - np.sum(Psd[ freqs <= -3*B - offset] * df)
    Pout2 = np.sum(Psd[ freqs >= B + offset] * df) - np.sum(Psd[ freqs >= 3*B + offset] * df)
    
    Pout = np.max([Pout1, Pout2])
    
    ACLR = 10*np.log10(Pout / Pin)
    
    return ACLR


def calcMSE(x, y):
    """
    Calculate Mean Squared Error (MSE) between x and y

    Parameters
    ----------
    x : np.array
       First signal for MSE calc 
    
    y : np.array
       Second signal for MSE calc
           
    Returns
    -------
    MSE : float
        Mean Squared Error (MSE) between x and y
        
    """
    
    MSE = np.mean(np.abs(x - y)**2)
    
    return MSE