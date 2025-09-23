# -*- coding: utf-8 -*-
"""
======================================================================
Function to process the experimental data for DPD
======================================================================
"""

import numpy as np
import torch as th
import matplotlib.pyplot as plt

import hdf5storage

from datetime               import datetime
from scipy.signal           import hilbert, firwin
from scipy.constants        import pi

from optic.comm.modulation  import modulateGray
from optic.comm.ofdm        import modulateOFDM, demodulateOFDM
from optic.comm.metrics     import fastBERcalc, calcEVM
from optic.dsp.core         import pnorm, finddelay, firFilter, clockSamplingInterp, signal_power

from dpd_mp                 import MP_filter
from torchUtils             import fitFilterNN

def save_OFDM(awg_sig, bitsTx, symbTx, paramOFDM, paramDPD = None):
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

    mode         = getattr(paramOFDM, "mode", "ARoF")
    awg_model    = getattr(paramOFDM, "awg_model", "Tektronix")
    awg_filepath = getattr(paramOFDM, "awg_filepath", None)
    tx_info_path = getattr(paramOFDM, "tx_info_path", None)

    fc  = getattr(paramOFDM, "fc", 3e9)
    bw  = getattr(paramOFDM, "bw", 100e6)
    scs = getattr(paramOFDM, "scs", 480e3)
    modOrder = getattr(paramOFDM, "modOrder", 64)
    Fawg     = getattr(paramOFDM, "Fawg", 16e9)

    date = datetime.today().strftime('%Y%m%d')
    
    if paramOFDM.DPD_active:
        DPD_model = paramDPD.model
        awg_filename = f"{mode}_{DPD_model}_DPD_signal_{fc/1e9}GHz_{modOrder}QAM_BW{bw/1e6}MHz_SCS{scs/1e6}MHz_{date}"
        
    else:
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
    tx_data["nfft"]     = paramOFDM.Nfft
    tx_data["cplen"]    = paramOFDM.G
    tx_data["nullIdx"]  = paramOFDM.nullCarriers
    tx_data["pilotIdx"] = paramOFDM.pilotCarriers
    tx_data["qamSym"]   = symbTx

    hdf5storage.savemat(tx_info_path + "\\tx_info_OFDM_" + awg_filename, tx_data, format = '7.3')
    
    
def tx_OFDM(paramOFDM, paramDPD = None):
    """
    Generate OFDM signal, with or without DPD, for experimental test.

    Parameters
    ----------
    paramOFDM : optic.utils.parameters object
        An object containing the parameters for OFDM modulation.
        - modType : string, optional. Type of the modulation format. Default is "QAM".

        - modOrder : int, optional. Number of constellation symbols. Default is 64.

        - numOFDMframes : int, optional. Number of OFDM frames of the generated signal. Default is 100.

        - Nfft : scalar, optional. Size of the FFT. Default is 512.
        
        - Ni : scalar, optional. Number of information subcarriers.

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

        - DPD_active : bool, optional. Flag to indicate if the DPD is activated.
        
    
    paramDPD : optic.utils.parameters object
        An object containing the parameters for DPD model. It's only valid if the DPD is active. 

    Returns
    -------
    symbTx : Complex-valued array representing the transmitted constellation symbols.
    
    sigTx : Complex-valued array representing the basebad OFDM signal.
    
    sigTx_RF : Real-valued array representing the RF OFDM signal.

    """
    
    # Modulation parameters
    modOrder       = getattr(paramOFDM, "modOrder", 64)
    modType        = getattr(paramOFDM, "modType", "qam")

    numOFDMframes  = getattr(paramOFDM, "numOFDMframes", 100)
    scs            = getattr(paramOFDM, "scs", 480e3)
    Nfft           = getattr(paramOFDM, "Nfft", 512)
    G              = getattr(paramOFDM, "G", 4)
    Ni             = getattr(paramOFDM, "Ni", 512)

    fc             = getattr(paramOFDM, "fc", 3e9)
    Fawg           = getattr(paramOFDM, "Fawg", 16e9)
    DPD_active     = getattr(paramOFDM, "DPD_active", False)
    awg_model      = getattr(paramOFDM, "awg_model", "Tektronix")
    saveTx         = getattr(paramOFDM, "saveTx", False)
    seed           = getattr(paramOFDM, "seed", 2)

    Rs  = scs * Nfft               # Symbols rate
    SpS = int(Fawg/Rs)   # Samples per symbol

    # Check for memory limits
    samples_per_frame = SpS * (Nfft + G)
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
    paramOFDM.pilot = pilotSymb
    
    # OFDM signal generation and RF modulation
    
    if DPD_active:
        SpS_DPD = paramDPD.SpS_DPD
        
        paramOFDM.SpS = SpS_DPD
        sigTx = pnorm(modulateOFDM(symbTx, paramOFDM))
        paramOFDM.SpS = SpS
        
        sigTx = apply_DPD(sigTx, paramDPD)
        P_DPD = signal_power(sigTx)
        
        Fs_DPD = Fawg * SpS_DPD / SpS
        sigTx = clockSamplingInterp(sigTx.reshape(-1, 1), Fs_DPD, Fawg).ravel()
        
        h_dpd = firwin(4096, 2*Rs, fs = Fawg)
        sigTx = firFilter(h_dpd, sigTx)
        sigTx = np.sqrt(P_DPD) * pnorm(sigTx)    
        
    else:
        paramOFDM.SpS = SpS    
        sigTx = pnorm(modulateOFDM(symbTx, paramOFDM))
    
    t        = np.arange(0, sigTx.size)*1/Fawg
    sigTx_RF = np.real(sigTx*np.exp(1j*2*pi*fc*t))
    sigTx_RF = pnorm(sigTx_RF)

    if saveTx:
        save_OFDM(sigTx_RF, bitsTx, symbTx, paramOFDM, paramDPD)

    return symbTx, sigTx, sigTx_RF

    
def rx_OFDM(sigRx_RF, sigTx, paramOFDM, lpf = False, bw = 75e6, plot = False):
    """
    Demodulates OFDM signal, for experimental test.

    Parameters
    ----------
    
    sigRx_RF : Complex-valued array representing the received OFDM signal from experimental data.

    sigTx    : Complex-valued array representing the basebad OFDM signal.
    
    paramOFDM : optic.utils.parameters object
        An object containing the parameters for OFDM modulation.
        - modType : string, optional. Type of the modulation format. Default is "QAM".

        - modOrder : int, optional. Number of constellation symbols. Default is 64.

        - numOFDMframes : int, optional. Number of OFDM frames of the generated signal. Default is 100.

        - Nfft : scalar, optional. Size of the FFT. Default is 512.
        
        - Ni : scalar, optional. Number of information subcarriers.

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

        - DPD_active : bool, optional. Flag to indicate if the DPD is activated.
    
    lpf : bool, optional. Flag to indicate if the signal is filtered after demodulation. Default is False.
    
    bw : float, optional. Bandwidth of the low pass filter after demodulation. Default is 75e6 Hz.
    
    plot : bool, optional. If True, the received signal spectrum is plotted after each stage of processing. Default is False.
    
    
    Returns
    -------
    symbRx : Complex-valued array representing the received constellation symbols.
    
    sigRx : Complex-valued array representing the demodulated basebad OFDM signal.
    
    """
    
    Nfft = getattr(paramOFDM, "Nfft", 512)
    G    = getattr(paramOFDM, "G", 4)
    scs  = getattr(paramOFDM, "scs", 480e3)
    fc   = getattr(paramOFDM, "fc", 3e9)
    Fawg = getattr(paramOFDM, "Fawg", 16e9)
    Fdso = getattr(paramOFDM, "Fdso", 16e9)
    SpS  = getattr(paramOFDM, "SpS", 32)
    
    # DC level extraction
    sigRx_RF -= np.mean(sigRx_RF)

    # Band-pass filtering
    numtaps = 4096
    Rs = scs * Nfft
    
    f1 = fc - 1.5*Rs
    f2 = fc + 1.5*Rs
    hbp_RF = firwin(numtaps, (f1, f2), pass_zero = 'bandpass', fs = Fdso)

    sigRx_RF = firFilter(hbp_RF, sigRx_RF)

    if plot:
        print("Spectrum after DC level extraction and BP filtering")
        plot_spec(sigRx_RF, Fawg, xlim = 10e9)

    # Resampling to AWG frequency
    if Fdso != Fawg:
        sigRx_RF = clockSamplingInterp(sigRx_RF.reshape(-1, 1), Fdso, Fawg).ravel()

    if plot:
        print("Spectrum after resampling from Fdso to Fawg")
        plot_spec(sigRx_RF, Fawg, xlim = 10e9)

    # Filtering to remove replicated spectrum from resampling
    f1 = fc - 1.5*Rs
    f2 = fc + 1.5*Rs
    hbp_RF = firwin(numtaps, (f1, f2), pass_zero = 'bandpass', fs = Fawg)

    sigRx_RF = firFilter(hbp_RF, sigRx_RF)

    if plot:
        print("Spectrum after filtering the replicated spectrum from resampling")
        plot_spec(sigRx_RF, Fawg, xlim = 10e9)

    # Downconversion
    t = np.arange(0, sigRx_RF.size)*1/Fawg
    sigRx = hilbert(sigRx_RF)*np.exp(-1j*2*pi*fc*t)

    if plot:
        print("Spectrum after downconversion")
        plot_spec(sigRx, Fawg, xlim = 0.25e9)

    sigRx = pnorm(sigRx)
    
    # Matching the array sizes for delay correction
    samples_per_frame = int(Fawg/Rs) * (Nfft + G)
    numOFDMframes_rx  = sigRx.size//samples_per_frame
    
    diff_samples = np.abs(sigRx.size - sigTx.size)

    if sigRx.size < sigTx.size:
        sigRx = np.pad(sigRx, (0, diff_samples))
    else:
        sigTx = np.pad(sigTx, (0, diff_samples))

    # Low pass filtering
    if lpf:
        hlp = firwin(numtaps, bw, fs = Fawg)
        sigRx = firFilter(hlp, sigRx)
        
        if plot:
            print("Spectrum after low pass filtering")
            plot_spec(sigRx, Fawg, xlim = 0.25e9)
    
    delay = finddelay(sigRx, sigTx)
    sigRx = np.roll(sigRx, -delay)
    
    # Phase correction
    samples_per_frame  = SpS * (Nfft + G)
    
    sigTx_par = np.reshape(sigTx[0:numOFDMframes_rx*samples_per_frame], (numOFDMframes_rx, samples_per_frame))        
    sigRx_par = np.reshape(sigRx[0:numOFDMframes_rx*samples_per_frame],  (numOFDMframes_rx, samples_per_frame))
    
    for frame in range(numOFDMframes_rx):
        rot   = np.mean(sigTx_par[frame,:]/sigRx_par[frame, :])
        sigRx_par[frame, :] = rot/np.abs(rot)*sigRx_par[frame, :]
    
    sigRx = sigRx_par.ravel()
        
    if plot:
        plot_sig([sigTx, sigRx], Fs = Fawg, labels = ["Tx", "Rx"], indx = np.arange(0, 10_000))

    # Decimation and symbols extraction
    symbRx_OFDM = sigRx.copy()[0::SpS][0:numOFDMframes_rx*(Nfft + G)]

    symbRx = demodulateOFDM(symbRx_OFDM, paramOFDM)
    symbRx = pnorm(symbRx)
    
    return symbRx, sigRx
    

def prepare_data_training(sigTx, sigRx, SpS_in, SpS_out, DPD_model, paramOFDM, device = "cpu"):
    Nfft = paramOFDM.Nfft
    G    = paramOFDM.G
    Fs   = paramOFDM.Fawg
    
    # Samples per frame before and after downsampling to SpS_out
    samples_per_frame  = SpS_in  * (Nfft + G)    
    numOFDMframes_rx = sigRx.size // samples_per_frame
    
    # Resampling of input and output signal
    sigRef = clockSamplingInterp(sigTx[0:numOFDMframes_rx*samples_per_frame].reshape(-1, 1), Fs, Fs/(SpS_in/SpS_out)).ravel()
    sigIn  = clockSamplingInterp(sigRx[0:numOFDMframes_rx*samples_per_frame].reshape(-1, 1), Fs, Fs/(SpS_in/SpS_out)).ravel()
    
    delay = finddelay(sigIn, sigRef)
    sigIn = np.roll(sigIn, -delay)
    
    rot = np.mean(sigRef/sigIn)
    sigIn = rot/np.abs(rot)*sigIn

    if DPD_model == "NN" or DPD_model == "KAN":
        sigRef = th.from_numpy(sigRef).to(device).type(th.complex64)
        sigIn  = th.from_numpy(sigIn).to(device).type(th.complex64)
    
    return sigRef, sigIn
    

def apply_DPD(sig, paramDPD):
    model = paramDPD.model
    DPD   = paramDPD.DPD
    
    if model == "NN" or model == "KAN":
        device  = paramDPD.device
        Ntaps   = paramDPD.Ntaps
        K       = paramDPD.K
        augment = paramDPD.augment
        
        sig = th.from_numpy(sig).to(device).type(th.complex64)
        
        DPD.eval()
        sig = fitFilterNN(sig, DPD, Ntaps, K, 1, 100, augment = augment)
        
        sig = sig.detach().cpu().numpy()
    
    else:
        P = paramDPD.P
        M = paramDPD.M
        
        sig = MP_filter(sig, np.conj(DPD).reshape((P, M)))

    return sig
    

def test_DPD_as_equalizer(sigRx, paramDPD, paramOFDM, lpf = False, bw = 75e6):
    Nfft = paramOFDM.Nfft
    G    = paramOFDM.G
    Fs   = paramOFDM.Fawg
    SpS  = paramOFDM.SpS
    SpS_DPD = paramDPD.SpS_DPD
    
    # Samples per frame before and after downsampling to SpS_out
    samples_per_frame  = SpS * (Nfft + G)    
    numOFDMframes_rx = sigRx.size // samples_per_frame
    
    sigRx_DPD = clockSamplingInterp(sigRx[0:numOFDMframes_rx*samples_per_frame].reshape(-1, 1), Fs, Fs/(SpS/SpS_DPD)).ravel()
    sigRx_DPD = apply_DPD(sigRx_DPD, paramDPD)
    sigRx_PA  = sigRx_DPD.copy()
    
    # Low-pass filtering
    if lpf:
        hlp = firwin(4096, bw, fs = Fs/(SpS/SpS_DPD))
        sigRx_DPD = firFilter(hlp, sigRx_DPD)
        sigRx_DPD -= np.mean(sigRx_DPD)
    
    # OFDM demodulation
    symbRx_OFDM = sigRx_DPD.copy()[0::SpS_DPD][0:numOFDMframes_rx*(Nfft + G)]
    symbRx_DPD = demodulateOFDM(symbRx_OFDM.copy(), paramOFDM)
    
    return symbRx_DPD, sigRx_PA


def calculate_metrics(symbTx, symbRx, discard, paramOFDM):
    """
    Calculated transmission metrics.

    Parameters
    ----------
    symbTx : Complex-valued array representing the transmitted constellation symbols.
    
    symbRx : Complex-valued array representing the received constellation symbols.
    
    discard : Int. Number of symbols to discard in metrics calculation.
    
    paramOFDM : optic.utils.parameters object
        An object containing the parameters for OFDM modulation.

    Returns
    ----------
    EVM : Error vector magnitude [%]
    
    BER : Bit error rate
    
    SNR : Signal to noise ratio [dB]
    
    """
    
    modOrder = paramOFDM.modOrder
    modType  = paramOFDM.modType
    Ni       = paramOFDM.Ni

    numOFDMframes_rx = symbRx.size // Ni
    
    index = np.arange(0, symbRx.size - discard)
    
    BER, _, SNR = fastBERcalc(symbRx[index], symbTx[0:Ni*numOFDMframes_rx][index], modOrder, modType)
    EVM = np.sqrt(calcEVM(symbRx[index], modOrder, modType, symbTx[0:Ni*numOFDMframes_rx][index]))*100

    return EVM, BER, SNR


def calcACLR(Psd, freqs, B):
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

    Returns
    -------
    float
        Calculated ACLR value in decibels (dB).

    Notes
    -----
    The ACLR measures the power leakage from one channel into an adjacent channel. 
    It is calculated as the ratio of the power outside of the adjacent channel 
    bandwidth to the power inside the adjacent channel bandwidth.

    The function computes the ACLR using the following steps:
    1. Compute the frequency resolution (df) as the difference between consecutive frequency values.
    2. Calculate the total power inside and outside the adjacent channel bandwidth.
    3. Compute the ratio of power outside to power inside the adjacent channel bandwidth.
    4. Convert the ratio to decibels (dB) using the `lin2dB` function.

    References
    ----------
    [1] 3GPP TS 36.101: "User Equipment (UE) radio transmission and reception."
        https://www.3gpp.org/ftp/Specs/html-info/36101.htm

    [2] 3GPP TS 38.104: "Base Station (BS) radio transmission and reception."
        https://www.3gpp.org/ftp/Specs/html-info/38104.htm
    """
    df = freqs[1] - freqs[0]
    Pin = np.sum(Psd[freqs >= -B] * df) - np.sum(Psd[freqs >= B] * df)
    Pout = np.sum(Psd[freqs <= -B] * df) + np.sum(Psd[freqs >= B] * df)

    return 10*np.log10(Pout / Pin)


# Plot functions

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