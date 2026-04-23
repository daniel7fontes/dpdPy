"""
================================================================
Utilities for DPD application (:mod:`dpd.utils`)
================================================================

   fitFilterNN  -- Apply the signal x to a NN model.
   filterMP     -- Apply the signal x to a MP model with coefficients w.
   applyDPD     -- Apply the signal to a DPD model processing.
   
"""

"""Utilities for DPD application."""


import contextlib
import numpy as np
import torch as th

from numba             import njit
from scipy.signal      import firwin
from optic.dsp.core    import clockSamplingInterp, signal_power, pnorm
from optic.dsp.coreGPU import firFilter

from dpd.train         import augmentFeatures


def fitFilterNN(x, model, paramModel, batchSize = 100, predict = True):
    """
    Apply a signal x to a NN model
    
    Parameters
    ----------
    x : np.array
        Complex-valued input signal
    
    model : object
        Neural network-based DPD model (ARVTDNN, ETDNN or ETDKAN)
    
    paramModel : optic.utils.parameters object
        An object containing the specification for model hyperparameters.
        - paramModel.M : int 
            Memory length of the model

        - paramModel.K : int 
            Maximum power order of the model (for ARVTDNN)
        
    batchSize : int
        Batch size for the signal division at model input (default is 100)
    
    predict : bool
        Flag that indicates whether the model is in predict mode (default is True)
        
    Returns
    -------
    y : th.tensor
        Output of the NN model
    
    """

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
    """
    Apply the signal x to a MP model with coefficients w
    
    Parameters
    ----------
    x : np.array
        Complex-valued input signal
    
    w : np.array
        Memory Polynomial array of coefficients
        
    M : int 
        Memory length of the model

    P : int 
        Maximum power order of the model
        
    Returns
    -------
    y : np.array
        Output of the MP model
    
    """
    
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


def applyDPD(sigTx, model, bw, Fs, Fs_DPD, paramModel):
    """
    Apply the signal to a DPD model processing
    
    Parameters
    ----------
    sigTx : np.array
        Complex-valued input signal
    
    model : object
        DPD model (MP, ARVTDNN, ETDNN or ETDKAN)
    
    bw : float
        RF bandwidth of sigTx
    
    Fs : float
        Sampling frequency of sigTx (Sa/s)
    
    Fs_DPD : float
        Sampling frequency of the signal for DPD
    
    paramModel : optic.utils.parameters object
        An object containing the specification for NN hyperparameters.
        - paramModel.M : int 
            Memory length of the model

        - paramModel.P : int 
            Maximum power order of the model (for MP)

        - paramModel.K : int 
            Maximum power order of the model (for ARVTDNN)
        
        - paramModel.hidden_layers : list
            Number of layers for each hidden layer (for ARVTDNN)
        
        - paramModel.activation : string
            Activation function for hidden layers (for ARVTDNN: "leaky_relu", "relu", "sigmoid", "tanh", "linear")
        
        - paramModel.N : int
            Size of the hidden layer (for ETDNN and ETDKAN)
        
        - paramModel.k : int
            B-spline polynomials order (for ETDKAN)
        
        - paramModel.grid : int
            B-spline grid (for ETDKAN)
        
        - paramModel.seed : int
            Seed for ETDKAN parameters initialization (for ETDKAN)
    
    Returns
    -------
    sigTx_DPD : np.array
        Normalized (in power) output of the DPD model
    
    gain_DPD : float
        Gain (in dB) of the DPD model
    
    """
    
    model_name = paramModel.model_name
    
    # Signal resampling from Fs to Fs_DPD
    sigTx = clockSamplingInterp(sigTx.reshape(-1, 1), Fs, Fs_DPD).ravel()
    
    # DPD application
    if model_name != "MP":
        model.eval()
        sigTx_DPD = fitFilterNN(th.from_numpy(sigTx.copy()).to(sigTx.device).type(th.complex64), \
                                model, paramModel, batchSize = 100, predict = True).detach().cpu().numpy()

    else:
        sigTx_DPD = filterMP(sigTx, model.w.ravel(), paramModel.M, paramModel.P)

    # Calc DPD gain
    gain_DPD  = 10*np.log10(signal_power(sigTx_DPD) / signal_power(sigTx))
    
    # Signal resampling from Fs_DPD to Fs, filtering and power normalization
    sigTx_DPD = clockSamplingInterp(sigTx_DPD.reshape(-1, 1), Fs_DPD, Fs).ravel()
    
    h_dpd = firwin(4096, 2*bw, fs = Fs)
    sigTx_DPD = firFilter(h_dpd, sigTx_DPD)
    
    sigTx_DPD = pnorm(sigTx_DPD)

    return sigTx_DPD, gain_DPD