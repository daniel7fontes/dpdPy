# -*- coding: utf-8 -*-
"""
Created on Wed Mar 18 10:54:17 2026

@author: PC
"""

import numpy as np

from scipy.constants import pi
from scipy.signal    import firwin, hilbert

from optic.models.channels    import linearFiberChannel
from optic.models.devices     import mzm, photodiode
from optic.dsp.core           import pnorm, signal_power
from optic.dsp.coreGPU        import firFilter
from optic.utils              import dBm2W


def RoF_channel(sigTx, paramRoF, filter_numtaps = 4096):
    
    """
    
    Calculates the output (after PA) signal of a ARoF system.
    
    Parameters
    ----------
    
    sigTx : np.array
        Input complex-valued signal
    
    
    paramRoF : optic.utils.parameters object
        Parameters for RoF channel.
        
        paramRoF.paramMZM : optic.utils.parameters object
            MZM parameters
            
            - paramRoF.paramMZM.Vpi : float
                - Half-wave voltage
            - paramRoF.paramMZM.Vb : float
                - Bias voltage
            - paramRoF.paramMZM.P_laser : float
                - Optical power at MZM input (dBm)
            - paramRoF.paramMZM.Pin_MZM : float
                - Electrical power of the RF signal at MZM input (dBm)
        
        paramRoF.paramRF : optic.utils.parameters object
            RF signal parameters
        
            - paramRoF.paramRF.fc_e : float
                - Frequency of the electrical carrier (Hz)
            - paramRoF.paramRF.bw : float
                - RF signal bandwidth
            - paramRoF.paramRF.Fs : float
                - Sampling frequency of the transmitted signal (Samples/s)
         
        paramRoF.paramFiber : optic.utils.parameters object
            Optical fiber parameters
        
            - paramRoF.paramFiber.L : float
                - Fiber length (km)
            - paramRoF.paramFiber.alpha : float
                - Fiber loss coefficient (dB/km)
            - paramRoF.paramFiber.D : float
                - Dispersion coefficient (ps/km.nm)
            - paramRoF.paramFiber.Fc : float
                - Optical carrier frequency (Hz)
            - paramRoF.paramFiber.Fs : float
                - Sampling frequency of the transmitted signal (Samples/s)
            
        paramRoF.paramPD : optic.utils.parameters object
            Photodetector parameters
        
            - paramRoF.paramPD.ideal : bool
                - Flag that indicates wheter the PD model is ideal (no band limitation and noise)
            - paramRoF.paramPD.B : float
                - PD bandwidth
            - paramRoF.paramPD.Ipd_sat : float
                - Saturation current
            - paramRoF.paramPD.Fs : float
                - Sampling frequency of the transmitted signal (Samples/s)
            
        paramRoF.paramPA : optic.utils.parameters object
            Power amplifier parameters
        
            - paramRoF.paramPA.model_name : string
                - Model name ("saleh", "rapp", "modified_rapp" or "limiter")
            
            (for Saleh model)
            - paramRoF.paramPA.alpha_a : float
            - paramRoF.paramPA.alpha_phi : float
            - paramRoF.paramPA.beta_a : float
            - paramRoF.paramPA.beta_phi : float
            
            (for Rapp model)
            - paramRoF.paramPA.g : float
            - paramRoF.paramPA.x_sat : float
            - paramRoF.paramPA.sigma_p : float
         
            (for modified Rapp model)
            - paramRoF.paramPA.g : float
            - paramRoF.paramPA.x_sat : float
            - paramRoF.paramPA.sigma_p : float
            - paramRoF.paramPA.alpha : float
            - paramRoF.paramPA.beta : float
            - paramRoF.paramPA.q : float
    
    filter_numtaps : int
        Number of taps for digital filters (default is 4096)
    
    """
    
    paramMZM     = paramRoF.paramMZM
    paramRF      = paramRoF.paramRF
    paramFiber   = paramRoF.paramFiber
    paramPD      = paramRoF.paramPD
    paramPA      = paramRoF.paramPA
    
    bw   = paramRF.bw
    Fs   = paramRF.Fs
    fc_e = paramRF.fc_e
    
    # 1 - Generating RF signal
    t = np.arange(0, len(sigTx))*1/Fs
    sigTx_RF = np.real( sigTx * np.exp(1j * 2*pi * fc_e * t) )
    gain_pre_MZM = 10**( (paramMZM.Pin_MZM - 10*np.log10(1e3*signal_power(sigTx_RF)) )/10)
    
    sigTx_RF = np.sqrt(gain_pre_MZM) * sigTx_RF
    sigTx_RF = np.clip(sigTx_RF, -paramMZM.Vpi/2, paramMZM.Vpi/2)

    # 2 - Optical modulation with MZM
    Ai     = np.sqrt(dBm2W(paramMZM.P_laser)) * np.ones(sigTx_RF.size)
    sigTxo = mzm(Ai, sigTx_RF, paramMZM)
    
    # 3 - Optical fiber propagation
    hopt_tx = firwin(filter_numtaps, fc_e + 2*bw, fs = Fs)
    sigTxo  = np.sqrt(signal_power(sigTxo)) * pnorm(firFilter(hopt_tx, sigTxo))
    
    sigRxo = linearFiberChannel(sigTxo, paramFiber)
        
    # 4 - Photodetection
    I_Rx = photodiode(sigRxo, paramPD)
    I_Rx -= I_Rx.mean()
    
    # 5 - Bandpass filter and demodulation
    hbp_RF = firwin(filter_numtaps, (fc_e - 2*bw, fc_e + 2*bw), pass_zero = 'bandpass', fs = Fs)
    I_RF = firFilter(hbp_RF, I_Rx)    
    sigRx = hilbert(I_RF)*np.exp(-1j * 2*pi * fc_e * t) * 1e3
    
    # 6 - Power amplifier    
    sigRx = power_amplifier(sigRx, paramPA)
    sigRx = pnorm(sigRx)
    
    return sigRx


def power_amplifier(x, paramPA):
    
    """
    Calculate the output of a specified PA model with a signal x at input
    
    Parameters
    ----------
    
    x : np.array
        Complex-valued signal at PA input
    
    paramPA : optic.utils.parameters object
        Power amplifier parameters
    
        - paramPA.model_name : string
            - Model name ("saleh", "rapp", "modified_rapp" or "limiter")
        
        (for Saleh model)
        - paramPA.alpha_a : float
        - paramPA.alpha_phi : float
        - paramPA.beta_a : float
        - paramPA.beta_phi : float
        
        (for Rapp model)
        - paramPA.g : float
        - paramPA.x_sat : float
        - paramPA.sigma_p : float
     
        (for modified Rapp model)
        - paramPA.g : float
        - paramPA.x_sat : float
        - paramPA.sigma_p : float
        - paramPA.alpha : float
        - paramPA.beta : float
        - paramPA.q : float
    
    Returns
    -------
    Output of the PA model
    
    """
    
    model_name = paramPA.model_name
    
    if model_name == "saleh":
        alpha_a   = paramPA.alpha_a
        alpha_phi = paramPA.alpha_phi
        beta_a    = paramPA.beta_a
        beta_phi  = paramPA.beta_phi
        
        return saleh(x, alpha_a, beta_a, alpha_phi, beta_phi)
    
    elif model_name == "rapp":
        g = paramPA.g
        x_sat = paramPA.x_sat
        sigma_p = paramPA.sigma_p
        
        return rapp(x, g, x_sat, sigma_p)
    
    elif model_name == "modified_rapp":
        g = paramPA.g
        x_sat = paramPA.x_sat
        sigma_p = paramPA.sigma_p
        alpha = paramPA.alpha
        beta = paramPA.beta 
        q = paramPA.q
        
        return modified_rapp(x, g, x_sat, sigma_p, alpha, beta, q)
    
    elif model_name == "limiter":
        x_sat = paramPA.x_sat
        y_sat = paramPA.y_sat
        
        return limiter(x, x_sat, y_sat)
        
    else:
        print("No model available")


def saleh(x, alpha_a = 2.1587, beta_a = 1.1517, alpha_phi = 4.033, beta_phi = 9.1040):
    
    abs_x = np.abs(x)
    
    G   = alpha_a / ( 1 + beta_a * abs_x**2 )
    Psi = (alpha_phi * abs_x**2) / (1 + beta_phi * abs_x**2)
    
    return G * np.exp(1j*Psi) * x

def rapp(x, g, x_sat, sigma_p):
    
    abs_x = np.abs(x)
    
    G = g / ( ( 1 + np.abs( abs_x / x_sat )**(2 * sigma_p) )**( 1 / (2*sigma_p) ) )
    
    return G * x
    
def modified_rapp(x, g = 16, x_sat = 1.9, sigma_p = 1.1, alpha = -345, beta = 0.17, q = 4):
    
    abs_x = np.abs(x)
    
    G = g / ( ( 1 + np.abs( g*abs_x / x_sat )**(2 * sigma_p) )**( 1 / (2*sigma_p) ) )
    Psi = (pi / 180) * ( alpha * abs_x**q ) / (1 + (abs_x/beta)**q )
    
    return G * np.exp(1j*Psi) * x
    

def limiter(x, x_sat, y_sat):
    g = y_sat / x_sat
    
    sat_points = np.where( np.abs(x) > x_sat )[0] 
    
    y = g*x
    
    if len(sat_points) != 0:
        y[sat_points] = y_sat * np.exp(1j*np.angle(y[sat_points]))
        
    return y
    