# -*- coding: utf-8 -*-
"""
======================================================================
Funções verificação de desempenho da DPD
======================================================================
"""

import numpy as np
import torch as th

from scipy.signal      import firwin
from optic.dsp.core    import clockSamplingInterp, signal_power, pnorm
from dpd.torchUtils    import fitFilterNN, filterMP
from optic.dsp.coreGPU import firFilter

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


def applyDPD(sigTx, model, Rs, Fs, Fs_DPD, paramTrain, paramModel):
    model_name = paramModel.model_name
    
    # Signal resampling from Fs to Fs_DPD
    sigTx = clockSamplingInterp(sigTx.reshape(-1, 1), Fs, Fs_DPD).ravel()
    
    if model_name != "MP":
        model.eval()
        sigTx_DPD = fitFilterNN(th.from_numpy(sigTx.copy()).to(paramTrain.device).type(th.complex64), \
                                model, paramTrain, paramModel, batchSize = 100, predict = True).detach().cpu().numpy()

    else:
        sigTx_DPD = filterMP(sigTx, model.w.ravel(), paramModel.M, paramModel.P)

    # Calc DPD gain
    gain_DPD  = 10*np.log10(signal_power(sigTx_DPD) / signal_power(sigTx))
    
    # Signal resampling from Fs to Fs_DPD
    h_dpd = firwin(4096, 2*Rs, fs = Fs)
    sigTx_DPD = clockSamplingInterp(sigTx_DPD.reshape(-1, 1), Fs_DPD, Fs).ravel()
    sigTx_DPD = firFilter(h_dpd, sigTx_DPD)
    sigTx_DPD = pnorm(sigTx_DPD)

    return sigTx_DPD, gain_DPD


def clip_complex(sig, max_amp):
    clip_pos = np.where( np.abs(sig) > max_amp )[0]
    
    if (len(clip_pos) != 0):
        for i in clip_pos:
            sig[i] = sig[i] * max_amp / np.abs(sig[i])
    
    return sig