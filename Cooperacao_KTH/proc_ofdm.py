# -*- coding: utf-8 -*-
"""
======================================================================
Function for processing of the experimental data for DPD
======================================================================
"""


import numpy as np
import matplotlib.pyplot as plt

import hdf5storage

from numpy.fft              import fft, fftshift
from datetime               import datetime
from scipy.signal           import hilbert, firwin
from scipy.interpolate      import interp1d
from scipy.constants        import pi

from optic.comm.modulation  import modulateGray
from optic.comm.ofdm        import modulateOFDM #, demodulateOFDM
from optic.dsp.core         import pnorm, finddelay, firFilter, clockSamplingInterp


def save_OFDM(awg_sig, bitsTx, symbTx, param):
    """
    Save transmitted OFDM signal.

    Parameters
    ----------
    param : optic.utils.parameters object
        An object containing the parameters for OFDM modulation.
        - awg_model : string, optional. Model of the AWG. Default is Tektronix.

        - awg_filepath : string. Location where the signal's file will be saved.

        - tx_info_path : string. Location where the signal's information file will be saved.

        - max_samples : int, optional. Maximum number of samples of the save signal. Default is 131072.

        - modOrder : int, optional. Number of constellation symbols. Default is 64.

        - fc : float, optional. Electrical carrier frequency. Default is 3e9 Hz.

        - bw : float, optional. Channel bandwidth. Default is 100e6 Hz.

        - scs : float, optional. Subcarriers spacing. Default is 480e3 Hz.

        - Fawg : float, optional. Sampling frequency of the AWG. Default is 16e9 Hz.

    """

    mode         = getattr(param, "mode", "ARoF")
    awg_model    = getattr(param, "awg_model", "Tektronix")
    awg_filepath = getattr(param, "awg_filepath", None)
    tx_info_path = getattr(param, "tx_info_path", None)

    fc  = getattr(param, "fc", 3e9)
    bw  = getattr(param, "bw", 100e6)
    scs = getattr(param, "scs", 480e3)
    modOrder = getattr(param, "modOrder", 64)
    Fawg     = getattr(param, "Fawg", 16e9)

    date = datetime.today().strftime('%Y%m%d')

    awg_filename = f"{mode}_signal_{fc/1e9}GHz_{modOrder}QAM_BW{bw/1e6}MHz_SCS{scs/1e6}MHz_{date}"

    if awg_model == "Tektronix":
        Waveform_M1_1 = np.zeros(awg_sig.size)
        Waveform_M1_1[0:int(10e3)] = 1

        mat_data = {}
        mat_data["Waveform_Name_1"]          = awg_filename
        mat_data["Waveform_Data_1"]          = awg_sig
        mat_data["Waveform_Sampling_Rate_1"] = Fawg
        mat_data["Waveform_M1_1"]            = Waveform_M1_1.astype(np.int8)

    elif awg_model == "R&S":
        mat_data = {}
        mat_data["I"]  = awg_sig.real
        mat_data["Q"]  = awg_sig.imag
        mat_data["fc"] = Fawg

    else:
        mat_data = {}
        mat_data["Y"] = awg_sig.real + 1j*awg_sig.imag
        mat_data["XDelta"] = 1/Fawg

    hdf5storage.savemat(awg_filepath + "\\" + awg_filename, mat_data, format='7.3', oned_as='column', store_python_metadata=True)

    # Save Tx info
    tx_data = {}
    tx_data["yctx_awg"] = awg_sig
    tx_data["data"]     = bitsTx
    tx_data["nfft"]     = param.Nfft
    tx_data["cplen"]    = param.G
    tx_data["nullIdx"]  = param.nullCarriers
    tx_data["pilotIdx"] = param.pilotCarriers
    tx_data["qamSym"]   = symbTx

    hdf5storage.savemat(tx_info_path + "\\tx_info_OFDM_" + awg_filename, tx_data, format = '7.3')
    
    
    
def tx_OFDM(param):
    """
    Generate OFDM signal.

    Parameters
    ----------
    param : optic.utils.parameters object
        An object containing the parameters for OFDM modulation.
        - modType : string, optional. Type of the modulation format. Default is "QAM".

        - modOrder : int, optional. Number of constellation symbols. Default is 64.

        - numOFDMframes : int, optional. Number of OFDM frames of the generated signal. Default is 100.

        - Nfft : scalar, optional. Size of the FFT. Default is 512.

        - G : scalar, optional. Cyclic prefix length. Default is 4.

        - hermitSymmetry : bool, optional. If True, indicates real OFDM symbols; if False, indicates complex OFDM symbols. Default is False.

        - pilot : complex-valued scalar, optional. Pilot symbol. Default is 1 + 1j.

        - pilotCarriers : np.array, optional. Indexes of pilot subcarriers. Default is an empty array.

        - nullCarriers : np.array, optional. Indexes of null subcarriers. Default is an empty array.

        - fc : float, optional. Electrical carrier frequency. Default is 3e9 Hz.

        - Fawg : float, optional. Sampling frequency of the AWG. Default is 16e9 Hz.

        - Fdso : float, optional. Sampling frequency of the AWG. Default is 16e9 Hz.

        - saveTx : bool, optional. If True, the signal will be saved in a .mat file. Default is False.

        - seed : int, optional. Determines the pattern of the random signal generated. Default is 2.

    Returns
    -------
    sigTx_RF : Real-valued array representing the OFDM signal in time-domain.

    symbTx : Complex-valued array representing the constellation symbols transmitted by the signal.

    """

    # Modulation parameters
    modOrder       = getattr(param, "modOrder", 64)
    modType        = getattr(param, "modType", "qam")

    numOFDMframes  = getattr(param, "numOFDMframes", 100)
    scs            = getattr(param, "scs", 480e3)
    Nfft           = getattr(param, "Nfft", 512)
    G              = getattr(param, "G", 4)
    hermitSymmetry = getattr(param, "hermitSymmetry", False)

    pilotCarriers  = getattr(param, "pilotCarriers", np.array([], dtype = np.int64))
    nullCarriers   = getattr(param, "nullCarriers", np.array([], dtype = np.int64))

    fc             = getattr(param, "fc", 3e9)
    Fawg           = getattr(param, "Fawg", 16e9)
    awg_model      = getattr(param, "awg_model", "Tektronix")
    saveTx         = getattr(param, "saveTx", False)
    seed           = getattr(param, "seed", 2)

    # Number of pilot, nulls and information carriers
    Np = len(pilotCarriers)
    Nz = len(nullCarriers)

    Ni = Nfft//2 - 1 - Np - Nz if hermitSymmetry else Nfft - Np - Nz

    Rs  = scs * Nfft     # Symbols rate
    param.SpS = int(Fawg/Rs)   # Samples per symbol

    # Check for memory limits
    samples_per_frame = param.SpS * (Nfft + G)
    samples_total     = samples_per_frame * numOFDMframes

    if awg_model == "Tektronix":
        if Fawg < 1.49e3 or Fawg > 50e9:
            raise ValueError("Fawg isn't in the range (1.49k - 50G Sa/s) of sampling frequency for the Tektronix AWG")

        if Fawg < 25e9 and samples_total > 8e9:
            raise ValueError(f"Number of samples ({samples_total}) exceeds the maximum (8_000_000_000).")

        if Fawg > 25e9 and samples_total > 16e9:
            raise ValueError(f"Number of samples ({samples_total}) exceeds the maximum (16_000_000_000).")


    # Bits generation and constellation symbols mapping
    np.random.seed(seed)
    bitsTx = np.random.randint(2, size = (numOFDMframes*Ni, int(np.log2(modOrder))))

    symbTx = modulateGray(bitsTx, modOrder, modType)
    symbTx = pnorm(symbTx)

    pilotSymb = 0.25*(max(symbTx.real) + 1j*max(symbTx.imag))
    param.pilot = pilotSymb

    sigTx_BB = pnorm(modulateOFDM(symbTx, param))
    t        = np.arange(0, sigTx_BB.size)*1/Fawg

    sigTx_RF = np.real(sigTx_BB*np.exp(1j*2*pi*fc*t))
    sigTx_RF = pnorm(sigTx_RF)

    if saveTx:
        save_OFDM(sigTx_RF, bitsTx, symbTx, param)

    return sigTx_RF, sigTx_BB, symbTx

    

def rx_OFDM(sigRx, sigTx_BB, param, mmse_eq = False, plot = False):
    """
    Demodulates OFDM signal.

    Parameters
    ----------
    sigRx : complex-valued array. Received RF OFDM signal.

    sigTx_BB : complex-valued array. Transmitted OFDM signal in baseband.

    param : optic.utils.parameters object
        An object containing the parameters for OFDM modulation.
        - modType : string, optional. Type of the modulation format. Default is "QAM".

        - modOrder : int, optional. Number of constellation symbols. Default is 64.

        - numOFDMframes : int, optional. Number of OFDM frames of the generated signal. Default is 100.

        - Nfft : scalar, optional. Size of the FFT. Default is 512.

        - G : scalar, optional. Cyclic prefix length. Default is 4.

        - hermitSymmetry : bool, optional. If True, indicates real OFDM symbols; if False, indicates complex OFDM symbols. Default is False.

        - pilot : complex-valued scalar, optional. Pilot symbol. Default is 1 + 1j.

        - pilotCarriers : np.array, optional. Indexes of pilot subcarriers. Default is an empty array.

        - nullCarriers : np.array, optional. Indexes of null subcarriers. Default is an empty array.

        - SpS : int, optional. Oversampling factor. Default is 2.

        - fc : float, optional. Electrical carrier frequency. Default is 3e9 Hz.

        - Fawg : float, optional. Sampling frequency of the AWG. Default is 16e9 Hz.

        - Fdso : float, optional. Sampling frequency of the AWG. Default is 16e9 Hz.

        - saveTx : bool, optional. If True, the signal will be saved in a .mat file. Default is False.

        - seed : int, optional. Determines the pattern of the random signal generated. Default is 2.

    Returns
    -------
    np.array or tuple
        If `returnChannel` is False, returns a complex-valued array representing the demodulated symbols sequence received.
        If `returnChannel` is True, returns a tuple containing the demodulated symbols sequence received and the estimated channel.

    """


    numOFDMframes = getattr(param, "numOFDMframes", 100)
    Nfft          = getattr(param, "Nfft", 512)
    G             = getattr(param, "G", 4)
    scs           = getattr(param, "scs", 480e3)
    fc            = getattr(param, "fc", 3e9)
    Fawg          = getattr(param, "Fawg", 16e9)
    Fdso          = getattr(param, "Fdso", 16e9)
    SpS_in        =  getattr(param, "SpS", 32)
    SpS_out       =  getattr(param, "SpS_out", 4)
    
    # DC level extraction
    sigRx -= np.mean(sigRx)

    # Band-pass filtering
    numtaps = 4096
    Rs = scs * Nfft

    f1 = fc - 1.5*Rs
    f2 = fc + 1.5*Rs
    hbp_RF = firwin(numtaps, (f1, f2), pass_zero = 'bandpass', fs = Fdso)

    sigRx = firFilter(hbp_RF, sigRx)

    if plot:
        print("Spectrum after DC level extraction and BP filtering")
        plot_spec(sigRx, Fawg, xlim = 10e9)

    # Resampling to AWG frequency
    if Fdso != Fawg:
        sigRx = clockSamplingInterp(sigRx.reshape(-1, 1), Fdso, Fawg).ravel()

    if plot:
        print("Spectrum after resampling from Fdso to Fawg")
        plot_spec(sigRx, Fawg, xlim = 10e9)

    # Filtering to remove replicated spectrum from resampling
    f1 = fc - 1.5*Rs
    f2 = fc + 1.5*Rs
    hbp_RF = firwin(numtaps, (f1, f2), pass_zero = 'bandpass', fs = Fawg)

    sigRx = firFilter(hbp_RF, sigRx)

    if plot:
        print("Spectrum after filtering the replicated spectrum from resampling")
        plot_spec(sigRx, Fawg, xlim = 10e9)

    # Downconversion
    t = np.arange(0, sigRx.size)*1/Fawg
    sigRx = hilbert(sigRx)*np.exp(-1j*2*pi*fc*t)

    if plot:
        print("Spectrum after downconversion")
        plot_spec(sigRx, Fawg, xlim = 0.25e9)

    sigRx = pnorm(sigRx)
    sigRx_BB = sigRx.copy() 
    
    # Delay correction
    samples_per_frame = int(Fawg/Rs) * (Nfft + G)
    numOFDMframes_rx = sigRx.size//samples_per_frame
    
    diff_samples = np.abs(sigRx.size - sigTx_BB.size)

    if sigRx.size < sigTx_BB.size:
        sigRx = np.pad(sigRx, (0, diff_samples))
    else:
        sigTx_BB = np.pad(sigTx_BB, (0, diff_samples))

    delay = finddelay(sigRx, sigTx_BB)
    sigRx = np.roll(sigRx, -delay)
    
    # Low pass filtering
    bw = 75e6
    hlp = firwin(numtaps, bw, fs = Fawg)
    sigRx = firFilter(hlp, sigRx)
    sigRx -= np.mean(sigRx)
    
    delay = finddelay(sigRx, sigTx_BB)
    sigRx = np.roll(sigRx, -delay)
    
    if plot:
        print("Spectrum after low pass filtering")
        plot_spec(sigRx, Fawg, xlim = 0.25e9)
    
    # Downsampling to SpS_out
    samples_per_frame_in  = SpS_in  * (Nfft + G)
    samples_per_frame_out = SpS_out * (Nfft + G)
    
    sigRx_BB = clockSamplingInterp(sigRx_BB[0:numOFDMframes_rx*samples_per_frame_in].reshape(-1, 1), Fawg, Fawg/(SpS_in/SpS_out)).ravel()
    sigRx    = clockSamplingInterp(sigRx[0:numOFDMframes_rx*samples_per_frame_in].reshape(-1, 1), Fawg, Fawg/(SpS_in/SpS_out)).ravel()
    sigRef   = clockSamplingInterp(sigTx_BB[0:numOFDMframes_rx*samples_per_frame_in].reshape(-1, 1), Fawg, Fawg/(SpS_in/SpS_out)).ravel()
    
    delay = finddelay(sigRx, sigRef)
    sigRx = np.roll(sigRx, -delay)
    
    # Phase correction and MMSE equalizer
    sigRef_par = np.reshape(sigRef[0:numOFDMframes_rx*samples_per_frame_out], (numOFDMframes_rx, samples_per_frame_out))        
    sigRx_par  = np.reshape(sigRx[0:numOFDMframes_rx*samples_per_frame_out],  (numOFDMframes_rx, samples_per_frame_out))
    
    for frame in range(numOFDMframes_rx):
        rot   = np.mean(sigRef_par[frame,:]/sigRx_par[frame, :])
        sigRx_par[frame, :] = rot/np.abs(rot)*sigRx_par[frame, :]
        
        h_mmse = []
        
        if mmse_eq:
            Ntaps = 5
            Nsamples = samples_per_frame_out
            
            print(f"\n- Frame {frame + 1}:")
            
            MSE = 10*np.log10(np.mean( np.abs(sigRx_par[frame,:] - sigRef_par[frame,:])**2 ))
            print(f"MSE before equalization = {MSE:.3f} dB")
            
            sigRx_par[frame, :], h = mmse_equalizer(sigRef_par[frame,:], sigRx_par[frame,:], Ntaps, Nsamples)
            
            delay = finddelay(sigRx_par[frame, :], sigRef_par[frame, :])
            sigRx_par[frame, :] = np.roll(sigRx_par[frame, :], -delay)
            
            MSE = 10*np.log10(np.mean( np.abs(sigRx_par[frame,:] - sigRef_par[frame,:])**2 ))
            print(f"MSE after equalization = {MSE:.3f} dB")
            
            h_mmse.append(h)

        sigRx = sigRx_par.ravel()
        sigRef = sigRef.ravel()
        
    if plot:
        plot_sig([sigRef, sigRx], Fs = Fawg / (SpS_in/SpS_out), labels = ["Tx", "Rx"], indx = np.arange(0, 500))

    # Decimation
    symbRx_OFDM = sigRx[0::SpS_out][0:numOFDMframes_rx*(Nfft + G)]

    if mmse_eq:
        symbRx = demodulateOFDM_v2(symbRx_OFDM, param)
        symbRx = pnorm(symbRx)

        return sigRx, sigRx_BB, symbRx, h_mmse
    
    else:
        symbRx = demodulateOFDM_v2(symbRx_OFDM, param)
        symbRx = pnorm(symbRx)

        return sigRx, sigRx_BB, symbRx
    


def estimate_correlation_matrix(x, N):
    M = len(x)
    if N > M:
        raise ValueError("Order N should be smaller than or equal to the length of the sequence M.")

    # Create a matrix where each row is a shifted version of the original sequence
    X = np.array([x[i:M-N+i+1] for i in range(N)])

    # Compute the unbiased correlation matrix
    R = (X @ np.conj(X.T)) / (M - np.arange(N)[:, None])

    return R

def estimate_cross_correlation(x, d, N):
    M = len(x)
    if len(d) != M:
        raise ValueError("The sequences x and d must have the same length.")
    if N > M:
        raise ValueError("N should be smaller than or equal to the length of the sequences.")

    p = np.zeros(N, dtype = complex)
    count = np.zeros(N)  # To keep track of the number of terms contributing to each element of p

    # Estimate the unbiased cross-correlation
    for k in range(N, M):  # Start from k = N to ensure we can form x_vec
        x_vec = x[k:k-N:-1]  # Create the vector x_vec = [x[k], x[k-1], ..., x[k-N+1]]
        p += x_vec * np.conj(d[k])
        count += 1  # Keep track of the number of terms contributing to each element

    # Normalize to make the estimation unbiased
    p /= count

    return p


def mmse_equalizer(x, y, Ntaps, Nsamples, sigma = 0):
    R = estimate_correlation_matrix(y[0:Nsamples], Ntaps)
    p = estimate_cross_correlation(y[0:Nsamples], x[0:Nsamples], Ntaps)

    h = (np.linalg.inv(R) + sigma * np.eye(Ntaps)) @ p
    h   /= np.linalg.norm(h)
    
    y_eq = firFilter(np.conj(h), y)

    return y_eq, h
    

def demodulateOFDM_v2(sig, param=None):
    """
    Demodulate OFDM signal.

    Parameters
    ----------
    sig : np.np.array
        Complex-valued array representing the OFDM signal sequence received at one sample per symbol.
    param : optic.utils.parameters object, optional
        Parameters for OFDM demodulation.

        - param.Nfft : scalar, optional. Size of the FFT [default: 512].
        - param.G : scalar, optional. Cyclic prefix length [default: 4].
        - param.hermitSymmetry : bool, optional. If True, indicates real OFDM symbols; if False, indicates complex OFDM symbols [default: False].
        - param.pilot : complex-valued scalar, optional. Pilot symbol [default: 1 + 1j].
        - param.pilotCarriers : np.array, optional. Indexes of pilot subcarriers [default: an empty array].
        - param.nullCarriers : np.array, optional. Indexes of null subcarriers [default: an empty array].
        - param.returnChannel : bool, optional. If True, return the estimated channel [default: False].

    Returns
    -------
    np.array or tuple
        If `returnChannel` is False, returns a complex-valued array representing the demodulated symbols sequence received.
        If `returnChannel` is True, returns a tuple containing the demodulated symbols sequence received and the estimated channel.

    Notes
    -----
    - The input signal must be sampled at one sample per symbol.
    - This function performs demodulation of the OFDM signal according to the provided parameters, including channel estimation and single tap equalization.

    References
    ----------
    [1] Proakis, J. G., & Salehi, M. Digital Communications (5th Edition). McGraw-Hill Education, 2008.
    """

    # Check and set default values for input parameters
    Nfft = getattr(param, "Nfft", 512)
    G = getattr(param, "G", 4)
    hermitSymmetry = getattr(param, "hermitSymmetry", False)
    pilot = getattr(param, "pilot", 0.25 + 0.25j)
    returnChannel = getattr(param, "returnChannel", False)
    pilotCarriers = getattr(param, "pilotCarriers", np.array([], dtype=np.int64))
    nullCarriers = getattr(param, "nullCarriers", np.array([], dtype=np.int64))

    Ns = Nfft // 2 - 1 if hermitSymmetry else Nfft
    Np = len(pilotCarriers)
    Nz = len(nullCarriers)
    Ni = Ns - Np - Nz

    Carriers = np.arange(0, Ns)
    dataCarriers = np.setdiff1d(Carriers, np.union1d(pilotCarriers, nullCarriers))

    numSymb = len(sig)

    if numSymb % (Nfft + G) != 0:
        raise ValueError(
            f"Number of received symbols ({numSymb}) is not divisible by Nfft + G ({Nfft + G})."
        )

    numOFDMframes = numSymb // (Nfft + G)

    H_abs = 0
    H_pha = 0

    sig_par = np.reshape(sig, (numOFDMframes, Nfft + G))

    # Cyclic prefix removal
    sig_par = sig_par[:, G : G + Nfft]

    # FFT operation
    for indFrame in range(numOFDMframes):
        sig_par[indFrame, :] = fftshift(fft(sig_par[indFrame, :])) / np.sqrt(Nfft)

    if hermitSymmetry:
        # Removal of hermitian symmetry
        sig_par = sig_par[:, 1 : 1 + Ns]

    # Channel estimation and single tap equalization
    if Np != 0:
        # Channel estimation
        for indFrame in range(numOFDMframes):
            H_est = sig_par[indFrame, :][pilotCarriers] / pilot

            H_abs += interp1d(
                pilotCarriers, np.abs(H_est), kind="linear", fill_value="extrapolate"
            )(Carriers)
            H_pha += interp1d(
                pilotCarriers, np.angle(H_est), kind="linear", fill_value="extrapolate"
            )(Carriers)

            if indFrame == numOFDMframes - 1:
                H_abs = H_abs / numOFDMframes
                H_pha = H_pha / numOFDMframes

        for indFrame in range(numOFDMframes):
            sig_par[indFrame, :] = sig_par[indFrame, :] / ( np.exp(1j * H_pha) )

    # Data carriers
    sig_par = sig_par[:, dataCarriers]

    if returnChannel:
        return sig_par.ravel(), H_abs * np.exp(1j * H_pha)
    else:
        return sig_par.ravel()

    
def plot_spec(sig, Fs, xlim):
    fig, axs = plt.subplots(figsize = (8, 4))
    axs.psd(sig, Fs = Fs/1e9, NFFT = 16*1024, color = "b", sides = 'twosided')
    axs.set_xlabel("f [GHz]")
    axs.set_ylabel("Power Spectral Density [dB/Hz]")
    axs.set_xlim(-xlim/1e9, xlim/1e9)
    axs.set_ylim(-40, 20)
    axs.set_yticks(np.arange(-40, 21, 10))
    axs.grid(True)
    plt.tight_layout()
    plt.show()
    

def plot_sig(sig, Fs, labels = ["Tx", "Rx"], indx = np.arange(0, 500)):
    fig, axs = plt.subplots(4, 1, figsize = (8, 16))
    indx = np.arange(1, 500)
    t = np.arange(0, sig[0].size)/(Fs)
    
    for i in range(len(sig)):
        axs[0].plot(t[indx]*1e9, np.real(sig[i])[indx], label = labels[i])
        axs[1].plot(t[indx]*1e9, np.imag(sig[i])[indx], label = labels[i])
        axs[2].plot(t[indx]*1e9, np.abs(sig[i])[indx],  label = labels[i])
        axs[3].psd(sig[i], Fs = Fs/1e9, NFFT = 16*1024, sides = 'twosided', label = labels[i])
        
    
    axs[0].set_xlabel("t [ns]")
    axs[0].set_ylabel("Real")
    axs[0].set_xlim(0, np.max(t[indx])*1e9)
    axs[0].grid(True)
    axs[0].legend(framealpha = 1, fontsize = 16, loc = "upper right")
    
    axs[1].set_xlabel("t [ns]")
    axs[1].set_ylabel("Imag")
    axs[1].set_xlim(0, np.max(t[indx])*1e9)
    axs[1].grid(True)
    axs[1].legend(framealpha = 1, fontsize = 16, loc = "upper right")
    
    axs[2].set_xlabel("t [ns]")
    axs[2].set_ylabel("Absolute")
    axs[2].set_xlim(0, np.max(t[indx])*1e9)
    axs[2].grid(True)
    axs[2].legend(framealpha = 1, fontsize = 16, loc = "upper right")

    axs[3].set_xlabel("f [GHz]")
    axs[3].set_ylabel("Power Spectral Density [dB/Hz]")
    axs[3].set_xlim(-0.25, 0.25)
    axs[3].set_ylim(-40, 20)
    axs[3].set_yticks(np.arange(-40, 21, 10))
    axs[3].grid(True)
    axs[3].legend(framealpha = 1, fontsize = 16, loc = "upper right")

    plt.tight_layout()
    plt.show()
    


def plot_const(symb, colors, index, save = False, show = False, filename = None):
    fig, axs = plt.subplots(figsize = (7, 7))
    
    for i in range(len(symb)):    
        axs.plot(symb[i][index].real, symb[i][index].imag, "o", color = colors[i], ms = 2)
    
    axs.set_ylabel("Quadrature - Q", fontsize = 20)
    axs.set_xlabel("In-phase - I", fontsize = 20)
    plt.axis("square")
    
    axs.set_xlim(-1.5, 1.5)
    axs.set_ylim(-1.5, 1.5)
    axs.minorticks_on()
    axs.tick_params(axis = 'both', top = "True", right = "True", which='minor',  width=1, direction = "in")
    axs.tick_params(axis = 'both', top = "True", right = "True", which='major',  width=1.5, direction = "in")
    
    plt.grid()
    plt.tight_layout()
    
    if save:
        plt.savefig(filename)
    
    if not(show):
        plt.close()