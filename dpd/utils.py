# -*- coding: utf-8 -*-
"""
======================================================================
Funções verificação de desempenho da DPD
======================================================================
"""

import contextlib
import numpy as np
import torch as th

from numba import njit

from scipy.signal      import firwin
from optic.dsp.core    import clockSamplingInterp, signal_power, pnorm
from optic.dsp.coreGPU import firFilter

from dpd.train         import augmentFeatures


def fitFilterNN(x, model, paramTrain, paramModel, batchSize = 100, predict = True):
    
    model_name = paramModel.model_name
    M = paramModel.M 
    
    if model_name == "ARVTDNN":
        K = paramModel.K
        augment = True
    else:
        K = 0
        augment = False
    
    xPad = th.nn.functional.pad(x, ((M+1)//2, (M+1)//2), "constant", 0)
    
    model.eval() if predict else model.train()
    dataSize   = len(x)
    numBatches = dataSize // batchSize
    
    indTaps = th.arange(0, (M + 1), dtype = th.int64)
    y = th.zeros(dataSize, dtype = th.complex64, device = x.device)

    if augment:
        xPad = augmentFeatures(xPad, K)
    else:
        xPad = th.view_as_real(xPad).to(th.float32)

    with th.no_grad() if predict else contextlib.nullcontext():
        for k in range(numBatches):
            start_idx = k * batchSize
            end_idx   = (k + 1) * batchSize
            
            sampleInd = th.arange(start_idx, end_idx, dtype=th.int64)
            indIn = (indTaps + sampleInd[:, None])  # Broadcasting to avoid nested loops

            x = xPad[indIn.flatten(), :].reshape(batchSize, -1)  # Flattening and reshaping
            
            y[sampleInd] = th.view_as_complex(model(x)).squeeze(0)
            
    return y


@njit
def filterMP(x, w, M, P):
    dataSize = x.size
    
    ind = np.arange(0, M + 1)    
    y = np.zeros(dataSize, dtype = np.complex128)

    x_window = np.zeros(2 * M + dataSize, dtype=np.complex128)
    for i in range(dataSize):
        x_window[i] = x[i]
        
    xk = np.zeros(P * (M + 1), dtype=np.complex128)

    for i in range(dataSize):
        X = x_window[i - ind]
        j = 0
        
        for p in range(P):
            for m in range(M + 1):
                xk[j] = X[m] * (np.abs(X[m]) ** p)
                j += 1
  
        y[i] = np.dot(xk, np.conj(w))

    return y


def applyDPD(sigTx, model, bw, Fs, Fs_DPD, paramTrain, paramModel):
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
    h_dpd = firwin(4096, 2*bw, fs = Fs)
    sigTx_DPD = clockSamplingInterp(sigTx_DPD.reshape(-1, 1), Fs_DPD, Fs).ravel()
    sigTx_DPD = firFilter(h_dpd, sigTx_DPD)
    sigTx_DPD = pnorm(sigTx_DPD)

    return sigTx_DPD, gain_DPD