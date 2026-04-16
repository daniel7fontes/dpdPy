# -*- coding: utf-8 -*-
"""
Created on Thu Aug  7 13:57:53 2025

@author: PC
"""

import numpy as np
from scipy.signal       import welch, firwin

from optic.comm.ofdm    import demodulateOFDM
from optic.comm.metrics import calcEVM
from optic.dsp.core     import pnorm, finddelay, clockSamplingInterp, decimate
from optic.dsp.coreGPU  import firFilter
from optic.utils        import parameters

from dpd.channel_models import RoF_channel
from dpd.train          import trainNN, trainMP
from dpd.utils          import applyDPD
from dpd.calc_metrics   import calcACLR


def get_pareto(f1, f2, n_trials):
    solutions = np.hstack( (f1.reshape((n_trials, 1)), f2.reshape((n_trials, 1))) )
    pareto = []
    
    for i, s in enumerate(solutions):
        test = np.where( (s[0] > solutions[:,0]) * (s[1] > solutions[:,1]) )[0]
        
        if not(len(test)):
            s_pareto = np.append(s, i+1)
            pareto.append(tuple(s_pareto))

    pareto.sort()
    pareto = np.array(pareto)

    pareto_trials = pareto[:,2].astype(np.int64)
    pareto_solutions = pareto[:,0:2]
    
    return pareto_solutions, pareto_trials    


def get_best_pareto(pareto_solutions, weights = (1, 1)):
    J1 = pareto_solutions[:,1]
    J2 = pareto_solutions[:,0]
    w1, w2 = weights
    
    num_sol = J1.size

    gamma_1 = 1/(w1*np.abs(np.max(J1)))
    gamma_2 = 1/(w2*np.abs(np.max(J2)))
    
    ideal = np.array([np.min(J1), np.min(J2)])
    
    distance_pareto = np.zeros(num_sol)
    for i in range(num_sol):
        distance_pareto[i] = np.sqrt( (gamma_1*(J1[i] - ideal[0]))**2 + (gamma_2*(J2[i] - ideal[1]))**2 )

    best_arg = np.argmin(distance_pareto)
    best = (J1[best_arg], J2[best_arg])
        
    return best, best_arg, ideal
    

def objective_rof_dpd(trial, data, paramOFDM, paramRoF, paramModel, paramTrain, paramMetrics):
    # Extract data parameters
    sigIn   = data.sigIn
    sigRef  = data.sigRef
    sigTx   = data.sigTx
    
    Rs        = data.Rs
    SpS       = data.SpS
    Fs        = data.Fs
    Fs_DPD    = data.Fs_DPD
    modOrder  = data.modOrder 
    constType = data.constType
    
    # Extract parameters for metrics calculation
    bw_for_aclr     = paramMetrics.bw_for_aclr
    offset_for_aclr = paramMetrics.offset_for_aclr
    discard         = paramMetrics.discard
    metrics         = paramMetrics.metrics
    model_path      = paramMetrics.model_path
    
    model_name = paramModel.model_name
    
    if model_name == "ARVTDNN":
        N1 = trial.suggest_int("N1", 5, 30)
        N2 = trial.suggest_int("N2", 5, 20)
        
        paramModel.hidden_layers = [N1, N2]
        paramModel.M = trial.suggest_int("M", 1, 5, step = 2)
        paramModel.K = trial.suggest_int("K", 1, 3)
        
        model, trainLoss, testLoss = trainNN(sigIn, sigRef, paramTrain, paramModel)
        model.save(model_path + f"\\{model_name}_trial_{trial.number + 1}.pth")
        
    elif model_name == "ETDNN":
        paramModel.M = trial.suggest_int("M", 1, 19, step = 2)
        paramModel.N = trial.suggest_int("N", 1, 20)
        
        model, trainLoss, testLoss = trainNN(sigIn, sigRef, paramTrain, paramModel)
        model.save(model_path + f"\\{model_name}_trial_{trial.number + 1}.pth")
        
    elif model_name == "ETDKAN":
        paramModel.M    = trial.suggest_int("M", 1, 3, step = 2)
        paramModel.N    = trial.suggest_int("N", 1, 4) 
        paramModel.k    = trial.suggest_int("k", 2, 6)
        paramModel.grid = trial.suggest_int("grid", 2, 6)
        
        model, trainLoss, testLoss = trainNN(sigIn, sigRef, paramTrain, paramModel)        
        model.save(model_path + f"\\{model_name}_trial_{trial.number + 1}")
    
    elif model_name == "MP":    
        paramModel.P = trial.suggest_int("P", 1, 10)
        paramModel.M = trial.suggest_int("M", 0, 10)
        paramTrain.S = 5e-2*np.eye(paramModel.P*(paramModel.M + 1), dtype = complex)
        
        model, trainLoss = trainMP(sigIn, sigRef, paramTrain, paramModel)
        model.save(model_path + f"\\{model_name}_trial_{trial.number + 1}.txt")
        
    else:
        print("DPD model not in the list")
    
    # Saving train/test losses
    np.savetxt(model_path + f"\\{model_name}_trainLoss_trial_{trial.number + 1}.txt", trainLoss, fmt = "%f")
    if model_name != "MP":
        np.savetxt(model_path + f"\\{model_name}_testLoss_trial_{trial.number + 1}.txt", testLoss, fmt = "%f")

    sigTx_DPD, gain_DPD = applyDPD(sigTx, model, Rs, Fs, Fs_DPD, paramTrain, paramModel)
    paramRoF.paramMZM.Pin_MZM = 17 + gain_DPD  # fix this
    
    sigRx_PA_DPD = RoF_channel(sigTx_DPD, paramRoF, filter_numtaps = 4096)
    
    hlp = firwin(4096, Rs/1.75, fs = Fs)
    sigRx_DPD = firFilter(hlp, sigRx_PA_DPD)
    
    delay = finddelay(sigRx_DPD, sigTx)
    sigRx_DPD = np.roll(sigRx_DPD, -delay)
    
    rot = np.mean(sigTx/sigRx_DPD)
    sigRx_DPD = rot/np.abs(rot)*sigRx_DPD
    
    # Decimation
    paramDec = parameters()
    paramDec.SpS_in  = SpS
    paramDec.SpS_out = 1
    
    symbRx_OFDM = decimate(sigRx_DPD, paramDec).ravel()
    symbRx_DPD  = demodulateOFDM(symbRx_OFDM, paramOFDM)
    
    # EVM, ACLR calculation
    index = np.arange(0, symbRx_DPD.size - discard)
    EVM   = np.sqrt(calcEVM(symbRx_DPD[index], modOrder, constType)[0])*100
    
    # Resampling from Fs to Fs_DPD for ACLR calc
    sigRx_PA_DPD = clockSamplingInterp(sigRx_PA_DPD.reshape(-1, 1), Fs, Fs_DPD).ravel()
    freq, P_sigRx_PA_DPD = welch(pnorm(sigRx_PA_DPD), fs = Fs_DPD, nfft = 16*1024, return_onesided = False)
    
    ACLR  = calcACLR(P_sigRx_PA_DPD, freq, bw_for_aclr, offset_for_aclr)
    NFLOP = model.calcNFLOP()
    
    full_metrics_array = np.array([EVM, ACLR, NFLOP]) if type(NFLOP) != list else np.concatenate( (np.array([EVM, ACLR]), np.array(NFLOP)) )
    
    np.savetxt(model_path + rf"\\{model_name}_metrics_{trial.number + 1}.txt", full_metrics_array, fmt = '%f')

    # Calculate only the metrics for Optuna trial output
    metrics_dic = {"EVM" : EVM, "ACLR" : ACLR, "NFLOP" : np.mean(np.array(NFLOP))}
    out = []
    
    for m in metrics_dic.keys():
        if m in metrics:
            out.append(metrics_dic[m])
    
    out = tuple(out)
        
    return out