# -*- coding: utf-8 -*-
"""
Created on Thu Aug  7 13:57:53 2025

@author: PC
"""

import numpy as np
import matplotlib.pyplot as plt

from scipy.signal           import welch, firwin, hilbert
from scipy.constants        import pi

from optic.comm.ofdm        import modulateOFDM, demodulateOFDM
from optic.models.channels  import linearFiberChannel
from optic.models.devices   import mzm, photodiode
from optic.comm.metrics     import calcEVM
from optic.dsp.core         import pnorm, signal_power, upsample, decimate, finddelay
from optic.dsp.coreGPU      import firFilter
from optic.utils            import parameters, dBm2W
from optic_private.dsp.core import calcACLR

from DPD_utils.MP_tools     import powerAmplifier, MP_filter, MP_training
from DPD_utils.NN_tools     import NN_training, KAN_training

import torch as th
from optic_private.torchUtils import fitFilterNN


def get_pareto(f1, f2, n_trials):
    
    solutions = np.hstack( (f1.reshape((n_trials, 1)), f2.reshape((n_trials, 1))) )
    pareto_solutions = []
    
    for i, s in enumerate(solutions):
        test = np.where( (s[0] > solutions[:,0]) * (s[1] > solutions[:,1]) )[0]
        
        if not(len(test)):
            pareto_solutions.append(tuple(s))

    pareto_solutions.sort()
    return np.array(pareto_solutions)


def RoF_channel(sigTx, paramRoF, paramOFDM):
    G_1, G_2, G_3 = paramRoF.G_list
    SpS = paramRoF.SpS
    Fs  = paramRoF.Fs
    Rs  = paramRoF.Rs 

    paramMZM = paramRoF.paramMZM
    paramRF  = paramRoF.paramRF
    paramChannel = paramRoF.paramChannel
    paramPD = paramRoF.paramPD
    
    # Sinal RF
    t = np.arange(0, len(sigTx))*1/Fs
    sigTx_RF = np.real( sigTx * np.exp(1j*2*pi*paramRF.fc_e*t) )
    sigTx_RF *= G_1
    
    # Sinal óptico
    Ai     = np.sqrt(dBm2W(paramMZM.Pin_OF))*np.ones(sigTx_RF.size)
    sigTxo = mzm(Ai, sigTx_RF, paramMZM)
    
    numtaps = 4096
    hopt_tx = firwin(numtaps, paramRF.fc_e + 2*Rs, fs = Fs)
    sigTxo = np.sqrt(signal_power(sigTxo))*pnorm(firFilter(hopt_tx, sigTxo))
    
    sigRxo = linearFiberChannel(sigTxo, paramChannel)
    
    # Sinal elétrico (fotocorrente)
    I_Rx = photodiode(sigRxo, paramPD)
    I_Rx -= I_Rx.mean()
    
    # Sinal elétrico pós-FPF
    numtaps = 4096
    f1 = paramRF.fc_e - 2*Rs
    f2 = paramRF.fc_e + 2*Rs
    hbp_RF = firwin(numtaps, (f1, f2), pass_zero = 'bandpass', fs = Fs)
    
    I_RF = firFilter(hbp_RF, I_Rx)
    
    # Sina elétrico pós-PA
    sigRx = hilbert(I_RF)*np.exp(-1j*2*pi*paramRF.fc_e*t)
    sigRx *= G_2
    
    sigRx = powerAmplifier(sigRx)
    
    sigRx_PA = sigRx.copy()
    sigRx_PA *= G_3
    
    numtaps = 4096
    #hlp = firwin(numtaps, Rs/1.5, fs = Fs)
    #sigRx = firFilter(hlp, sigRx)
    
    delay = finddelay(sigRx, sigTx)
    sigRx = np.roll(sigRx, -delay)
    
    rot = np.mean(sigTx/sigRx)
    sigRx = rot/np.abs(rot)*sigRx

    # Parâmetros da decimação
    paramDec = parameters()
    paramDec.SpS_in  = SpS
    paramDec.SpS_out = 1
    
    #symbRx_OFDM = decimate(sigRx, paramDec).ravel()
    symbRx_OFDM = sigRx[0::SpS]
    symbRx = demodulateOFDM(symbRx_OFDM, paramOFDM)
    
    return sigRx_PA, symbRx


def test_as_DPD(DPD_model, attr_test):
    symbTx    = attr_test.symbTx
    paramRoF  = attr_test.paramRoF
    paramOFDM = attr_test.paramOFDM
    paramDPD  = attr_test.paramDPD
    SpS_DPD   = paramDPD.SpS_DPD

    # Generating OFDM signal
    paramOFDM.SpS = SpS_DPD
    sigTx = modulateOFDM(symbTx, paramOFDM)
    sigTx = pnorm(sigTx)
    
    # DPD    
    if not(paramDPD.DPD == "MP"):
        sigTx = th.from_numpy(sigTx).to(paramDPD.device).type(th.complex64)
        
        DPD_model.eval()
        sigTx_DPD = fitFilterNN(sigTx, DPD_model, paramDPD.Ntaps, paramDPD.K, 1, 1000, augment = paramDPD.augment)
        sigTx_DPD = sigTx_DPD.detach().cpu().numpy()
    
    else: 
        sigTx_DPD = MP_filter(sigTx, np.conj(DPD_model).reshape((paramDPD.P, paramDPD.M)))
        
    P_DPD = signal_power(sigTx_DPD)
    
    # Upsampling and filtering
    h_dpd = firwin(4096, 2*paramRoF.Rs, fs = paramRoF.Fs)
    sigTx_DPD = upsample(sigTx_DPD.reshape(-1,1), paramRoF.SpS//SpS_DPD).ravel()
    sigTx_DPD = firFilter(h_dpd, sigTx_DPD)
    sigTx_DPD = np.sqrt(P_DPD)*pnorm(sigTx_DPD)

    # Channel
    sigRx_PA_DPD, symbRx_DPD = RoF_channel(sigTx_DPD, paramRoF, paramOFDM)

    return sigRx_PA_DPD, symbRx_DPD


def objective_DPD(trial, attr_test, attr_train, metrics, model_path):
    
    # 
    sigIn     = attr_train.sigIn
    sigRef    = attr_train.sigRef
    paramDPD  = attr_test.paramDPD
    paramRoF  = attr_test.paramRoF
    paramOFDM = attr_test.paramOFDM
    
    SpS_DPD   = paramDPD.SpS_DPD
    
    
    if not(paramDPD.DPD == "MP"):
        if paramDPD.KAN:
            Nlayers = trial.suggest_int("Nlayers", 1, 2)
            N1      = trial.suggest_int("N1", 1, 20)
            N2      = trial.suggest_int("N2", 1, 20)
            
            layers_full = [N1, N2] 
            
            if not(paramDPD.directLearn):
                paramDPD.Ntaps = trial.suggest_int("Ntaps", 2, 10, step = 2)
            
            paramDPD.layers = layers_full[0:Nlayers]
            paramDPD.layers.append(2)
            paramDPD.layers.insert(0, 2*paramDPD.Ntaps)
    
            paramDPD.k = trial.suggest_int("k", 2, 5)
            paramDPD.grid = trial.suggest_int("grid", 3, 8)
            
            DPD_model, trainLoss, testLoss = KAN_training(sigIn[0:paramDPD.N], sigRef[0:paramDPD.N], paramDPD)
            DPD_model.saveckpt(model_path + f"\\KAN_model_trial{trial.number+1}")
            
            NFLOPs = 0
        
        else:
            # Optimized param
            Nlayers = trial.suggest_int("Nlayers", 1, 3)
            N1      = trial.suggest_int("N1", 2, 50)
            N2      = trial.suggest_int("N2", 2, 50)
            N3      = trial.suggest_int("N3", 2, 50)
            
            layers_full = [N1, N2, N3]
            
            if not(paramDPD.directLearn):
                paramDPD.Ntaps = trial.suggest_int("Ntaps", 2, 10, step = 2)
                
            paramDPD.K      = trial.suggest_int("K", 1, 5)
            paramDPD.layers = layers_full[0:Nlayers]
            paramDPD.layers.append(2)
            paramDPD.layers.insert(0, (paramDPD.K+2)*paramDPD.Ntaps)
            
            DPD_model, trainLoss, testLoss = NN_training(sigIn[0:paramDPD.N], sigRef[0:paramDPD.N], paramDPD)
            th.save(DPD_model.state_dict(), model_path + f"\\NN_model_trial{trial.number+1}.pth")
            
            NFLOPs = 0
            for i in range(Nlayers + 1):
                NFLOPs += 2*paramDPD.layers[i] + paramDPD.layers[i+1]*(2*paramDPD.layers[i] - 1)
            NFLOPs += 2 + paramDPD.Ntaps*(5 - paramDPD.K)
            
            
    else:
        paramDPD.P = trial.suggest_int("P", 1, 8)
        paramDPD.M = trial.suggest_int("M", 1, 12)
        #lbd = trial.suggest_float("lbd", 0.9, 0.99999)
        paramDPD.S = 5e-2*np.eye(paramDPD.P*paramDPD.M, dtype = complex)
        
        DPD_model, _, __, trainLoss = MP_training(sigRef, paramDPD, sigIn)
        NFLOPs = paramDPD.M*(11*paramDPD.P + 6) - 1
        
    # Test as DPD
    sigRx_PA_DPD, symbRx_DPD = test_as_DPD(DPD_model, attr_test)
    
    # EVM, MSE, ACLR
    discard = paramOFDM.Ni
    index = np.arange(0, symbRx_DPD.size - discard)
        
    freq, P_sigRx_PA_DPD = welch(pnorm(sigRx_PA_DPD)[0::paramRoF.SpS//SpS_DPD], fs = SpS_DPD*paramRoF.Rs, nfft = 16*1024, return_onesided = False)
    ACLR = calcACLR(P_sigRx_PA_DPD, freq, 0.5e9)
    EVM = np.sqrt(calcEVM(symbRx_DPD[index], paramOFDM.modOrder, paramOFDM.modType)[0])*100

    metrics_dic = {"EVM":EVM, "ACLR":ACLR, "NFLOPs":NFLOPs}
    out = []
    
    for m in metrics_dic.keys():
        if m in metrics:
            out.append(metrics_dic[m])
    
    out = tuple(out)
        
    return out