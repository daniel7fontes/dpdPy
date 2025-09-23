# -*- coding: utf-8 -*-
"""
Created on Thu Aug  7 13:57:53 2025

@author: PC
"""

import numpy as np
import torch as th

from scipy.signal           import welch, firwin, hilbert
from scipy.constants        import pi

from optic.comm.ofdm        import modulateOFDM, demodulateOFDM
from optic.models.channels  import linearFiberChannel
from optic.models.devices   import mzm, photodiode
from optic.comm.metrics     import calcEVM
from optic.dsp.core         import pnorm, signal_power, upsample, finddelay
from optic.dsp.coreGPU      import firFilter
from optic.utils            import parameters, dBm2W
from optic_private.dsp.core import calcACLR
from optic_private.torchUtils import fitFilterNN

from dpd.mp                 import MP_training, MP_filter, CMP_filter, LS_CMP_solver
from dpd.nn                 import NN_training, KAN_training
from dpd.utils              import power_amplifier


def get_pareto(f1, f2, n_trials):
    
    solutions = np.hstack( (f1.reshape((n_trials, 1)), f2.reshape((n_trials, 1))) )
    pareto_solutions = []
    pareto_trials = []
    
    for i, s in enumerate(solutions):
        test = np.where( (s[0] > solutions[:,0]) * (s[1] > solutions[:,1]) )[0]
        
        if not(len(test)):
            pareto_solutions.append(tuple(s))
            pareto_trials.append(i)

    pareto_solutions.sort()
    
    return np.array(pareto_solutions), np.array(pareto_trials)


def RoF_channel(sigTx, gain_DPD, paramRoF, paramOFDM):
    gain_pre_MZM, gain_pre_PA = paramRoF.G_list
    SpS = paramRoF.SpS
    Fs  = paramRoF.Fs
    Rs  = paramRoF.Rs 
    IQ_imb = paramRoF.IQ_imb
    
    paramMZM = paramRoF.paramMZM
    paramRF  = paramRoF.paramRF
    paramChannel = paramRoF.paramChannel
    paramPD = paramRoF.paramPD
    
    # Sinal RF
    t = np.arange(0, len(sigTx))*1/Fs
    
    sigTx_RF = np.real( sigTx * gain_DPD * np.exp(1j*2*pi*paramRF.fc_e*t) )
    sigTx_RF *= gain_pre_MZM
    sigTx_RF = np.clip(sigTx_RF, -paramMZM.Vpi/2, paramMZM.Vpi/2)

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
    sigRx = sigRx.real + 1j*sigRx.imag * IQ_imb
    sigRx *= gain_pre_PA
    
    sigRx = power_amplifier(sigRx)
    
    sigRx_PA = sigRx.copy()
    sigRx_PA = pnorm(sigRx_PA)
    
    numtaps = 4096
    hlp = firwin(numtaps, Rs/1.75, fs = Fs)
    sigRx = firFilter(hlp, sigRx)
    
    delay = finddelay(sigRx, sigTx)
    sigRx = np.roll(sigRx, -delay)
    
    rot = np.mean(sigTx/sigRx)
    sigRx = rot/np.abs(rot)*sigRx

    # Parâmetros da decimação
    paramDec = parameters()
    paramDec.SpS_in  = SpS
    paramDec.SpS_out = 1
    
    symbRx_OFDM = sigRx[0::SpS]
    symbRx = demodulateOFDM(symbRx_OFDM, paramOFDM)
    
    return sigRx_PA, symbRx


def test_as_dpd(DPD_model, attr_test):
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
    if (paramDPD.DPD == "NN" or paramDPD.DPD == "KAN" or paramDPD.DPD == "ENN" or paramDPD.DPD == "EKAN"):
        sigTx = th.from_numpy(sigTx).to(paramDPD.device).type(th.complex64)
        
        DPD_model.eval()
        sigTx_DPD = fitFilterNN(sigTx, DPD_model, paramDPD.Ntaps, paramDPD.K, 1, 1000, augment = paramDPD.augment)
        sigTx_DPD = sigTx_DPD.detach().cpu().numpy()
    
    elif (paramDPD.DPD == "MP"):
        sigTx_DPD = MP_filter(sigTx, np.conj(DPD_model).reshape((paramDPD.P, paramDPD.M)))
    
    else:
        sigTx_DPD = CMP_filter(sigTx, DPD_model[0], DPD_model[1])
        
    gain_DPD = np.sqrt(signal_power(sigTx_DPD))
    
    h_dpd = firwin(4096, 2*paramRoF.Rs, fs = paramRoF.Fs)
    sigTx_DPD = upsample(sigTx_DPD.reshape(-1,1), paramRoF.SpS//SpS_DPD).ravel()
    sigTx_DPD = firFilter(h_dpd, sigTx_DPD)
    sigTx_DPD = pnorm(sigTx_DPD)
    
    # Channel
    sigRx_PA_DPD, symbRx_DPD = RoF_channel(sigTx_DPD, gain_DPD, paramRoF, paramOFDM)

    return sigRx_PA_DPD, symbRx_DPD


def objective_dpd(trial, attr_test, attr_train, metrics, model_path):
    
    # 
    sigIn     = attr_train.sigIn
    sigRef    = attr_train.sigRef
    paramDPD  = attr_test.paramDPD
    paramRoF  = attr_test.paramRoF
    paramOFDM = attr_test.paramOFDM
    
    SpS_DPD   = paramDPD.SpS_DPD
    
    if paramDPD.DPD == "NN":
        # Optimized param
        Nlayers = trial.suggest_int("Nlayers", 1, 2)
        N1      = trial.suggest_int("N1", 2, 50)
        N2      = trial.suggest_int("N2", 2, 50)
        
        layers_full = [N1, N2]
        
        paramDPD.layers = layers_full[0:Nlayers]
        paramDPD.layers.append(2)
        
        if not(paramDPD.directLearn):
            paramDPD.Ntaps = trial.suggest_int("Ntaps", 2, 10, step = 2)
        
        if paramDPD.augment:
            paramDPD.K = trial.suggest_int("K", 1, 5)
            paramDPD.layers.insert(0, (2+paramDPD.K)*paramDPD.Ntaps)
        else:
            paramDPD.layers.insert(0, 2*paramDPD.Ntaps)
        
        DPD_model, trainLoss, testLoss = NN_training(sigIn[0:paramDPD.N], sigRef[0:paramDPD.N], paramDPD)
        th.save(DPD_model.state_dict(), model_path + f"\\NN_model_trial{trial.number+1}.pth")
        
        NFLOPs = 0
        
        K = paramDPD.K
        Ntaps = paramDPD.Ntaps 
        layers = paramDPD.layers
        
        for i in range(len(layers)-1):
            NFLOPs += 2*layers[i] * layers[i+1] 
            
            if i < (len(layers)-2):
                NFLOPs += layers[i+1]
        
        if paramDPD.augment:
            NFLOPs += 10*Ntaps + (K - 1) * Ntaps

    elif paramDPD.DPD == "ENN":
        # Optimized param
        Nlayers = 1
        N1 = trial.suggest_int("N1", 5, 50)
        
        layers_full = [N1]
        
        paramDPD.layers = layers_full[0:Nlayers]
        
        paramDPD.Ntaps = trial.suggest_int("Ntaps", 2, 20, step = 2)    
        paramDPD.layers.append(paramDPD.Ntaps)
        paramDPD.layers.insert(0, paramDPD.Ntaps)
        
        DPD_model, trainLoss, testLoss = NN_training(sigIn[0:paramDPD.N], sigRef[0:paramDPD.N], paramDPD)
        th.save(DPD_model.state_dict(), model_path + f"\\ENN_model_trial{trial.number+1}.pth")
        
        NFLOPs = 0
        
        Ntaps  = paramDPD.Ntaps 
        layers = paramDPD.layers
        
        for i in range(len(layers)-1):
            
            if i < (len(layers)-2):
                NFLOPs += 2*layers[i] * layers[i+1] 
                NFLOPs += layers[i+1]
                
            else:
                NFLOPs += 2 * ( 2*layers[i] * layers[i+1] )
        
        NFLOPs += 10*Ntaps + Ntaps*6 + 2*(Ntaps - 1)
        
    
    elif paramDPD.DPD == "KAN":
        Nlayers = trial.suggest_int("Nlayers", 0, 1)
        N1      = trial.suggest_int("N1", 2, 5)
        
        layers_full = [N1]
        
        if not(paramDPD.directLearn):
            paramDPD.Ntaps = trial.suggest_int("Ntaps", 2, 6, step = 2)
        
        paramDPD.layers = layers_full[0:Nlayers] if Nlayers != 0 else []
        paramDPD.layers.append(2)
        
        if paramDPD.augment:
            paramDPD.K = trial.suggest_int("K", 1, 5)
            paramDPD.layers.insert(0, (2+paramDPD.K)*paramDPD.Ntaps)
        else:
            paramDPD.layers.insert(0, 2*paramDPD.Ntaps)

        paramDPD.k = trial.suggest_int("k", 2, 5)
        paramDPD.grid = trial.suggest_int("grid", 2, 5)
        
        NFLOPs = 0
        b_flops = 10
        
        k = paramDPD.k
        grid = paramDPD.grid
        layers = paramDPD.layers
        
        for i in range(len(layers) - 1):
            NFLOPs += layers[i] * layers[i+1] * ( 9*k*(grid + 1.5*k) + 2*grid - 2.5*k + 3 ) + layers[i]*b_flops    

        if paramDPD.augment:
            NFLOPs += 10*paramDPD.Ntaps + (paramDPD.K - 1) * paramDPD.Ntaps
            
        DPD_model, trainLoss, testLoss = KAN_training(sigIn[0:paramDPD.N], sigRef[0:paramDPD.N], paramDPD)
        DPD_model.saveckpt(model_path + f"\\KAN_model_trial{trial.number+1}")
    
    
    elif paramDPD.DPD == "EKAN":
        Nlayers = 1
        N1      = trial.suggest_int("N1", 2, 5)
        
        layers_full = [N1]
        
        paramDPD.Ntaps = trial.suggest_int("Ntaps", 2, 10, step = 2)
        paramDPD.layers = [paramDPD.Ntaps, N1, paramDPD.Ntaps]
        
        paramDPD.k = trial.suggest_int("k", 2, 5)
        paramDPD.grid = trial.suggest_int("grid", 2, 5)
        
        NFLOPs = 0
        b_flops = 10
        
        k = paramDPD.k
        grid = paramDPD.grid
        layers = paramDPD.layers
        
        for i in range(len(layers) - 1):
            NFLOPs += layers[i] * layers[i+1] * ( 9*k*(grid + 1.5*k) + 2*grid - 2.5*k + 3 ) + layers[i]*b_flops    

        NFLOPs += 10*paramDPD.Ntaps + paramDPD.Ntaps*6 + 2*(paramDPD.Ntaps - 1)
        
        DPD_model, trainLoss, testLoss = KAN_training(sigIn[0:paramDPD.N], sigRef[0:paramDPD.N], paramDPD)
        #DPD_model.saveckpt(model_path + f"\\EKAN_model_trial{trial.number+1}")
    
    
    elif paramDPD.DPD == "MP":    
        paramDPD.P = trial.suggest_int("P", 1, 10)
        paramDPD.M = trial.suggest_int("M", 1, 12)
        paramDPD.S = 5e-2*np.eye(paramDPD.P*paramDPD.M, dtype = complex)
        
        DPD_model, _, __, trainLoss = MP_training(sigRef, paramDPD, sigIn)
        NFLOPs = paramDPD.M*(11*paramDPD.P + 6) - 1
        
        np.savetxt(model_path + f"\\MP_model_trial{trial.number+1}.txt", DPD_model, fmt = '%f')
    
    else:
        paramDPD.P1 = trial.suggest_int("P1", 1, 8)
        paramDPD.M1 = trial.suggest_int("M1", 1, 10)
        paramDPD.P2 = trial.suggest_int("P2", 1, 8)
        paramDPD.M2 = trial.suggest_int("M2", 1, 10)
        
        DPD_model = LS_CMP_solver(sigIn[0:paramDPD.N], sigRef[0:paramDPD.N], paramDPD.P1, paramDPD.P2, paramDPD.M1, paramDPD.M2)
        
        NFLOPs = (paramDPD.M1*(11*paramDPD.P1 + 6) - 1) + (paramDPD.M2*(11*paramDPD.P2 + 6) - 1)
        np.savetxt(model_path + f"\\CMP_model_1_trial{trial.number+1}.txt", DPD_model[0], fmt = '%f')
        np.savetxt(model_path + f"\\CMP_model_2_trial{trial.number+1}.txt", DPD_model[1], fmt = '%f')
        
            
    # Test as DPD
    sigRx_PA_DPD, symbRx_DPD = test_as_dpd(DPD_model, attr_test)
    
    # EVM, MSE, ACLR
    discard = paramOFDM.Ni
    index = np.arange(0, symbRx_DPD.size - discard)
        
    freq, P_sigRx_PA_DPD = welch(pnorm(sigRx_PA_DPD)[0::paramRoF.SpS//SpS_DPD], fs = SpS_DPD*paramRoF.Rs, nfft = 16*1024, return_onesided = False)
    ACLR = calcACLR(P_sigRx_PA_DPD, freq, 0.5e9)
    EVM = np.sqrt(calcEVM(symbRx_DPD[index], paramOFDM.modOrder, paramOFDM.modType)[0])*100
    
    # save metrics array    
    metrics_array = np.array([EVM, ACLR, NFLOPs])
    np.savetxt(model_path + f"\\metrics_{trial.number+1}.txt", metrics_array, fmt = '%f')
    
    metrics_dic = {"EVM":EVM, "ACLR":ACLR, "NFLOPs":NFLOPs}
    out = []
    
    for m in metrics_dic.keys():
        if m in metrics:
            out.append(metrics_dic[m])
    
    out = tuple(out)
        
    return out