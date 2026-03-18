# -*- coding: utf-8 -*-
"""
======================================================================
Funções verificação de desempenho da DPD
======================================================================
"""

import numpy as np
from optic.comm.metrics import calcEVM
from scipy.constants import pi


def calcMSE(x, y):
    """
    Estimativa do Erro Médio Quadrático entre os sinais de entrada x e de saída y
    
    Parameters
    ----------
    x : np.array
        Sinal de entrada do sistema
    y : np.array
        Sinal de saída do sistema
        
    Returns
    -------
    MSE : float
          Erro médio quadrático entre x e y [dB]
    """
    
    MSE = np.mean(np.abs(y - x)**2)
    return 10*np.log10(MSE)


def calcNMSE(x, y):
    """
    Estimativa do Erro Médio Quadrático Normalizado entre os sinais de entrada x e de saída y
    
    Parameters
    ----------
    x : np.array
        Sinal de entrada do sistema
    y : np.array
        Sinal de saída do sistema
        
    Returns
    -------
    NMSE : float
           Erro médio quadrático normalizado entre x e y [dB]
    """
    
    NMSE = np.mean(np.abs(y - x)**2) / np.mean(np.abs(x)**2)
    return 10*np.log10(NMSE)


def calcPAPR(signal):
    peak_power = np.max(np.abs(signal)) ** 2
    average_power = np.mean(np.abs(signal) ** 2)
    
    papr = peak_power / average_power
    
    return 10 * np.log10(papr)

def calcACLR(Psd, freqs, B, offset):
    """
    Calculate the Adjacent Channel Leakage Ratio (ACLR).

    Parameters
    ----------
    Psd : numpy.ndarray
        Power spectral density (Psd) values.
    freqs : numpy.ndarray
        Frequency values corresponding to the Psd array.
    B : float
        Bandwidth of the adjacent channel.
    offset : float
             frequency offset to start adjacent channel  

    Returns
    -------
    float
        Calculated ACLR value in decibels (dB)
    """
    df = freqs[1] - freqs[0]
    
    Pin = np.sum(Psd[freqs >= - B] * df) - np.sum(Psd[freqs >= B] * df)
    
    Pout1 = np.sum(Psd[ freqs <= -B - offset] * df) - np.sum(Psd[ freqs <= -3*B - offset] * df)
    Pout2 = np.sum(Psd[ freqs >= B + offset] * df) - np.sum(Psd[ freqs >= 3*B + offset] * df)

    Pout = np.max([Pout1, Pout2])
    
    return 10*np.log10(Pout / Pin)


def clip_complex(sig, max_amp):
    clip_pos = np.where( np.abs(sig) > max_amp )[0]
    
    if (len(clip_pos) != 0):
        for i in clip_pos:
            sig[i] = sig[i] * max_amp / np.abs(sig[i])
    
    return sig