# -*- coding: utf-8 -*-
"""
Created on Thu Aug  7 13:57:53 2025

@author: PC
"""

import numpy as np
import torch as th
from torch import nn

from scipy.signal             import welch, firwin, hilbert
from scipy.constants          import pi

from optic.comm.ofdm          import modulateOFDM, demodulateOFDM
from optic.models.channels    import linearFiberChannel
from optic.models.devices     import mzm, photodiode
from optic.comm.metrics       import calcEVM
from optic.dsp.core           import pnorm, signal_power, finddelay, clockSamplingInterp, decimate
from optic.dsp.coreGPU        import firFilter
from optic.utils              import parameters, dBm2W

from dpd.torchUtils           import fitFilterNN, MLP, ETDKAN, ETDNN
from dpd.mp                   import MP_training, MP_filter
from dpd.nn                   import NN_training, KAN_training
from dpd.utils                import power_amplifier, calcACLR


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


def get_best_pareto(J1, J2, w = 1):
    num_sol = J1.size

    gamma_1 = 1/np.abs(np.max(J1))
    gamma_2 = 1/np.abs(np.max(J2))
    
    ideal = np.array([np.min(J1), np.min(J2)])
    
    distance_pareto = np.zeros(num_sol)
    for i in range(num_sol):
        distance_pareto[i] = np.sqrt( (gamma_1*w*(J1[i] - ideal[0]))**2 + (gamma_2*(J2[i] - ideal[1]))**2 )

    best_arg = np.argmin(distance_pareto)
    best = (J1[best_arg], J2[best_arg])
        
    return best, best_arg, ideal


def RoF_channel(sigTx, paramRoF):
    paramOFDM    = paramRoF.paramOFDM
    paramMZM     = paramRoF.paramMZM
    paramRF      = paramRoF.paramRF
    paramChannel = paramRoF.paramChannel
    paramPD      = paramRoF.paramPD
    
    SpS = paramOFDM.SpS
    Rs  = paramOFDM.Rs
    Fs  = paramOFDM.Fs
    gain_pre_MZM, gain_pre_PA = paramOFDM.gain_pre_MZM_PA
    
    # Sinal RF
    t = np.arange(0, len(sigTx))*1/Fs
    
    sigTx_RF = np.real( sigTx * np.exp(1j*2*pi*paramRF.fc_e*t) )
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
    f1 = paramRF.fc_e - 2*Rs
    f2 = paramRF.fc_e + 2*Rs
    hbp_RF = firwin(numtaps, (f1, f2), pass_zero = 'bandpass', fs = Fs)
    
    I_RF = firFilter(hbp_RF, I_Rx)
    
    # Sinal elétrico pós-PA
    sigRx = hilbert(I_RF)*np.exp(-1j*2*pi*paramRF.fc_e*t)
    sigRx = sigRx.real + 1j*sigRx.imag
    sigRx *= gain_pre_PA
    
    sigRx = power_amplifier(sigRx)
    
    sigRx_PA = sigRx.copy()
    sigRx_PA = pnorm(sigRx_PA)
    
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
    
    symbRx_OFDM = sigRx.copy()[0::SpS][0:paramOFDM.numOFDMframes*(paramOFDM.Nfft + paramOFDM.G)]
    #symbRx_OFDM = clockSamplingInterp(sigRx.reshape(-1, 1), paramOFDM.Fs, paramOFDM.Fs/SpS).ravel()[0:paramOFDM.numOFDMframes*(paramOFDM.Nfft + paramOFDM.G)]

    #symbRx_OFDM = decimate(sigRx.copy(), paramDec).ravel()
    symbRx = demodulateOFDM(symbRx_OFDM, paramOFDM)
    
    return sigRx_PA, symbRx


def test_as_dpd(DPD, symbTx, paramRoF, paramDPD):
    
    # Generating OFDM signal
    paramOFDM = paramRoF.paramOFDM
    
    SpS = paramOFDM.SpS
    Fs  = paramOFDM.Fs
    SpS_DPD = paramDPD.SpS_DPD
    
    paramOFDM.SpS = SpS_DPD
    
    sigTx = modulateOFDM(symbTx, paramOFDM)
    sigTx = pnorm(sigTx)
    
    paramOFDM.SpS = SpS
    
    # DPD
    if (paramDPD.model == "MP"):
        sigTx_DPD = MP_filter(sigTx, np.conj(DPD).reshape((paramDPD.P, paramDPD.M)))
    
    else:
        sigTx = th.from_numpy(sigTx).to(paramDPD.device).type(th.complex64)
        
        DPD.eval()
        sigTx_DPD = fitFilterNN(sigTx, DPD, paramDPD.Ntaps, paramDPD.K, 1, 100, augment = paramDPD.augment)
        sigTx_DPD = sigTx_DPD.detach().cpu().numpy()
    
    gain_DPD = np.sqrt(signal_power(sigTx_DPD))
    
    h_dpd = firwin(4096, 2*paramOFDM.Rs, fs = paramOFDM.Fs)
    sigTx_DPD = clockSamplingInterp(sigTx_DPD.reshape(-1, 1), Fs*SpS_DPD/SpS, Fs).ravel()
    sigTx_DPD = firFilter(h_dpd, sigTx_DPD)
    sigTx_DPD = gain_DPD*pnorm(sigTx_DPD)
    
    # Channel
    sigRx_PA_DPD, symbRx_DPD = RoF_channel(sigTx_DPD, paramRoF)

    return sigRx_PA_DPD, symbRx_DPD


def getTrialParam(trial, param_list, file_path):
    param_values = []
    
    for i, pp in enumerate(param_list):
        param_values.append(np.loadtxt(file_path + f"\\{pp}.txt")[trial])
        print(f"{pp} = {param_values[i]}")
        
    return param_values
    

def calcNFLOPs(paramDPD):
    model  = paramDPD.model
    NFLOPs = 0
    
    if model == "ARVTDNN":
        layers = paramDPD.layers
        Ntaps  = paramDPD.Ntaps
        K      = paramDPD.K
        
        for i in range(len(layers) - 1):
            NFLOPs += 2*layers[i] * layers[i+1] 
            
            if i < (len(layers) - 2):
                NFLOPs += layers[i+1]
        
        NFLOPs += 10*Ntaps + (K - 1) * Ntaps
    
    elif model == "RVTDNN":
        layers = paramDPD.layers
        Ntaps  = paramDPD.Ntaps
        
        for i in range(len(layers) - 1):
            NFLOPs += 2*layers[i] * layers[i+1] 
            
            if i < (len(layers) - 2):
                NFLOPs += layers[i+1]
                
    elif model == "ETDNN":
        layers = paramDPD.layers
        Ntaps  = paramDPD.Ntaps 

        for i in range(len(layers)-1):
            
            if i < (len(layers)-2):
                NFLOPs += 2*layers[i] * layers[i+1] 
                NFLOPs += layers[i+1]
                
            else:
                NFLOPs += 2 * ( 2*layers[i] * layers[i+1] )

        NFLOPs += 10*Ntaps + 6*Ntaps + 2*(Ntaps - 1)

    elif model == "RVTDKAN":
        b_flops = 10
        
        layers = paramDPD.layers        
        Ntaps  = paramDPD.Ntaps
        k      = paramDPD.k
        grid   = paramDPD.grid
        
        for i in range(len(layers) - 1):
            NFLOPs += layers[i] * layers[i+1] * ( 9*k*(grid + 1.5*k) + 2*grid - 2.5*k + 3 ) + layers[i]*b_flops    

    elif model == "ETDKAN":
        b_flops = 10

        layers = paramDPD.layers    
        Ntaps  = paramDPD.Ntaps
        k      = paramDPD.k
        grid   = paramDPD.grid
        symb   = paramDPD.symb
        
        if symb:
            NFLOPs_estimation = {
            "sin"  : (10, 100),
            "cos"  : (10, 100),
            "tanh" : (10, 100),
            "sinh" : (10, 100),
            "exp"  : (10, 100),
            "abs"  : (10, 15),
            "x"    : (2, 7),
            "x^2"  : (3, 8),
            "x^3"  : (4, 9),
            "x^4"  : (5, 10),
            "x^5"  : (6, 11) 
            }
        
            NFLOPs_max, NFLOPs_min = np.zeros(2)
            
            for l in range(2):
                if l == 0:
                    i_sz = layers[1]
                    j_sz = Ntaps
                else:
                    i_sz = 2*Ntaps
                    j_sz = layers[1]
                
                for i in range(i_sz):
                    for j in range(j_sz):
                        func = paramDPD.DPD.KAN.symbolic_fun[l].funs_name[i][j]
                        NFLOPs_max += NFLOPs_estimation[func][1]
                        NFLOPs_min += NFLOPs_estimation[func][0]
            
            NFLOPs_min += 10*Ntaps + 6*Ntaps + 2*(Ntaps - 1)
            NFLOPs_max += 10*Ntaps + 6*Ntaps + 2*(Ntaps - 1)
            
            NFLOPs = (NFLOPs_min, NFLOPs_max)
            
        else:
            for i in range(len(layers) - 1):
                NFLOPs += layers[i] * layers[i+1] * ( 9*k*(grid + 1.5*k) + 2*grid - 2.5*k + 3 ) + layers[i]*b_flops    
    
            NFLOPs += 10*Ntaps + 6*Ntaps + 2*(Ntaps - 1)
    
    else:
        NFLOPs = paramDPD.M*(11*paramDPD.P + 6) - 1
        
    return NFLOPs


def objective_dpd(trial, train_data, paramRoF, paramDPD, metrics, model_path):
    
    sigIn   = train_data.sigIn
    sigRef  = train_data.sigRef
    symbTx  = train_data.symbTx 
    SpS_DPD = paramDPD.SpS_DPD
    
    if paramDPD.model == "ARVTDNN":
        # Optimized param
        Nlayers = trial.suggest_int("Nlayers", 1, 2)
        N1      = trial.suggest_int("N1", 5, 50)
        N2      = trial.suggest_int("N2", 2, 50)
        Ntaps   = trial.suggest_int("Ntaps", 2, 10, step = 2)
        K       = trial.suggest_int("K", 1, 3)
        
        layers_full = [N1, N2]
        
        paramDPD.layers = layers_full[0:Nlayers]
        paramDPD.layers.append(2)
        paramDPD.layers.insert(0, (2+K)*Ntaps)
        paramDPD.Ntaps = Ntaps
        paramDPD.K = K
        
        DPD, _, _ = NN_training(sigIn[0:paramDPD.N], sigRef[0:paramDPD.N], paramDPD)
        th.save(DPD.state_dict(), model_path + f"\\ARVTDNN_model_trial{trial.number}.pth")

    elif paramDPD.model == "RVTDNN":
        # Optimized param
        Nlayers = trial.suggest_int("Nlayers", 1, 2)
        N1      = trial.suggest_int("N1", 5, 50)
        N2      = trial.suggest_int("N2", 2, 50)
        Ntaps   = trial.suggest_int("Ntaps", 2, 10, step = 2)
                
        layers_full = [N1, N2]
        
        paramDPD.layers = layers_full[0:Nlayers]
        paramDPD.layers.append(2)
        paramDPD.layers.insert(0, 2*Ntaps)
        paramDPD.Ntaps = Ntaps
                
        DPD, _, _ = NN_training(sigIn[0:paramDPD.N], sigRef[0:paramDPD.N], paramDPD)
        th.save(DPD.state_dict(), model_path + f"\\RVTDNN_model_trial{trial.number}.pth")

    elif paramDPD.model == "ETDNN":
        # Optimized param
        N1    = trial.suggest_int("N1", 5, 50)
        Ntaps = trial.suggest_int("Ntaps", 2, 20, step = 2)    
        
        paramDPD.layers = [Ntaps, N1, Ntaps]
        paramDPD.Ntaps = Ntaps
        
        DPD, _, _ = NN_training(sigIn[0:paramDPD.N], sigRef[0:paramDPD.N], paramDPD)
        th.save(DPD.state_dict(), model_path + f"\\ETDNN_model_trial{trial.number}.pth")
        
    elif paramDPD.model == "RVTDKAN":
        Nlayers = trial.suggest_int("Nlayers", 0, 1)
        N1      = trial.suggest_int("N1", 2, 5)
        Ntaps   = trial.suggest_int("Ntaps", 2, 6, step = 2)
        k       = trial.suggest_int("k", 2, 5)
        grid    = trial.suggest_int("grid", 2, 5)
        
        layers_full = [N1]
        
        paramDPD.layers = layers_full[0:Nlayers] if Nlayers != 0 else []
        paramDPD.layers.append(2)
        paramDPD.layers.insert(0, 2*Ntaps)
        
        paramDPD.k    = k
        paramDPD.grid = grid
        
        DPD, _, _ = KAN_training(sigIn[0:paramDPD.N], sigRef[0:paramDPD.N], paramDPD)
        DPD.saveckpt(model_path + f"\\RVTDKAN_model_trial{trial.number}")
    
    elif paramDPD.model == "ETDKAN":
        Nlayers = 1
        N1      = trial.suggest_int("N1", 1, 4)
        Ntaps   = trial.suggest_int("Ntaps", 2, 4, step = 2)
        k       = trial.suggest_int("k", 2, 6)
        grid    = trial.suggest_int("grid", 2, 6)
        
        paramDPD.Ntaps  = Ntaps
        paramDPD.layers = [Ntaps, N1, 2*Ntaps]
        paramDPD.k      = k
        paramDPD.grid   = grid
        
        DPD, _, _ = KAN_training(sigIn[0:paramDPD.N], sigRef[0:paramDPD.N], paramDPD)
        
        if paramDPD.symb:
            DPD.set_symb()
            y_symb, y_func = DPD.get_symb()
            y_symb_str = [str(y_symb[i]) for i in range(len(y_symb)) ]
                
            file_name = model_path + f"\\ETDKAN_model_trial{trial.number}.txt"

            with open(file_name, "w") as f:
                for func in y_symb_str:
                    f.write(func+"\n"+"\n")
                f.close()
                
            func_act = []
            
            for l in range(2):
                if l == 0:
                    i_sz = paramDPD.layers[1]
                    j_sz = paramDPD.Ntaps
                else:
                    i_sz = 2*paramDPD.Ntaps
                    j_sz = paramDPD.layers[1]
                
                for i in range(i_sz):
                    for j in range(j_sz):
                        func_act.append(DPD.KAN.symbolic_fun[l].funs_name[i][j])
            
            file_name_func_act = model_path + f"\\ETDKAN_func_act_trial{trial.number}.txt"

            with open(file_name_func_act, "w") as f:
                for func in func_act:
                    f.write(func+"\n")
                f.close()

        else: 
            DPD.KAN.saveckpt(model_path + f"\\ETDKAN_model_trial{trial.number}")
    
    elif paramDPD.model == "MP":    
        paramDPD.P = trial.suggest_int("P", 1, 10)
        paramDPD.M = trial.suggest_int("M", 1, 12)
        paramDPD.S = 5e-2*np.eye(paramDPD.P*paramDPD.M, dtype = complex)
        
        DPD, _, _, _ = MP_training(sigRef, paramDPD, sigIn)
        
        np.savetxt(model_path + f"\\MP_model_trial{trial.number}.txt", DPD, fmt = '%f')
    
    else:
        print("DPD model not in the list")
    
    # Test as DPD
    paramDPD.DPD = DPD
    sigRx_PA_DPD, symbRx_DPD = test_as_dpd(DPD, symbTx, paramRoF, paramDPD)
    
    # EVM, ACLR calculation
    discard = 500
    index = np.arange(0, symbRx_DPD.size - discard)
    paramOFDM = paramRoF.paramOFDM
    
    freq, P_sigRx_PA_DPD = welch(pnorm(sigRx_PA_DPD)[0::paramOFDM.SpS//SpS_DPD], fs = SpS_DPD*paramOFDM.Rs, nfft = 16*1024, return_onesided = False)
    
    ACLR = calcACLR(P_sigRx_PA_DPD, freq, paramOFDM.bw/2, 2.5e6)
    EVM  = np.sqrt(calcEVM(symbRx_DPD[index], paramOFDM.modOrder, paramOFDM.modType)[0])*100
    NFLOPs = calcNFLOPs(paramDPD)
    
    # save metrics array
    if paramDPD.model == "ETDKAN":
        if DPD.symb:
            metrics_array = np.array([EVM, ACLR, NFLOPs[0], NFLOPs[1]])
            np.savetxt(model_path + f"\\metrics_{trial.number}.txt", metrics_array, fmt = '%f')
            NFLOPs = np.mean(NFLOPs)
        else:
            metrics_array = np.array([EVM, ACLR, NFLOPs])
            np.savetxt(model_path + f"\\metrics_{trial.number}.txt", metrics_array, fmt = '%f')
    else:
        metrics_array = np.array([EVM, ACLR, NFLOPs])
        np.savetxt(model_path + f"\\metrics_{trial.number}.txt", metrics_array, fmt = '%f')

    metrics_dic = {"EVM":EVM, "ACLR":ACLR, "NFLOPs":NFLOPs}
    out = []
    
    for m in metrics_dic.keys():
        if m in metrics:
            out.append(metrics_dic[m])
    
    out = tuple(out)
        
    return out


def test_dpd(trial, symbTx, paramRoF, paramDPD, study_path):
    
    SpS_DPD = paramDPD.SpS_DPD
    
    # Import DPD models
    if paramDPD.model == "ARVTDNN":
        N1      = np.loadtxt(study_path + r"\parameters\N1.txt").astype(np.int64)[trial]
        N2      = np.loadtxt(study_path + r"\parameters\N2.txt").astype(np.int64)[trial]
        Nlayers = np.loadtxt(study_path + r"\parameters\Nlayers.txt").astype(np.int64)[trial]
        Ntaps   = np.loadtxt(study_path + r"\parameters\Ntaps.txt").astype(np.int64)[trial]
        K       = np.loadtxt(study_path + r"\parameters\K.txt").astype(np.int64)[trial]
        
        layers_full = [N1, N2]
        
        paramDPD.layers = layers_full[0:Nlayers]
        paramDPD.layers.append(2)
        paramDPD.layers.insert(0, (2+K)*Ntaps)
        paramDPD.Ntaps = Ntaps
        paramDPD.K = K
        
        DPD = MLP(paramDPD.layers, activation = nn.ReLU() ).to(paramDPD.device)
        
        DPD.load_state_dict(th.load(study_path + rf"\models\ARVTDNN_model_trial{trial}.pth", weights_only = True))
        
    elif paramDPD.model == "RVTDNN":
        N1      = np.loadtxt(study_path + r"\parameters\N1.txt").astype(np.int64)[trial]
        N2      = np.loadtxt(study_path + r"\parameters\N2.txt").astype(np.int64)[trial]
        Nlayers = np.loadtxt(study_path + r"\parameters\Nlayers.txt").astype(np.int64)[trial]
        Ntaps   = np.loadtxt(study_path + r"\parameters\Ntaps.txt").astype(np.int64)[trial]
        
        layers_full = [N1, N2]
        
        paramDPD.layers = layers_full[0:Nlayers]
        paramDPD.layers.append(2)
        paramDPD.layers.insert(0, 2*Ntaps)
        paramDPD.Ntaps = Ntaps
        
        DPD = MLP(paramDPD.layers, activation = nn.ReLU() ).to(paramDPD.device)
        
        DPD.load_state_dict(th.load(study_path + rf"\models\RVTDNN_model_trial{trial}.pth", weights_only = True))
        
        
    elif paramDPD.model == "ETDNN":
        # Optimized param        
        N1    = np.loadtxt(study_path + r"\parameters\N1.txt").astype(np.int64)[trial]
        Ntaps = np.loadtxt(study_path + r"\parameters\Ntaps.txt").astype(np.int64)[trial]
        
        paramDPD.layers = [Ntaps, N1, Ntaps]
        paramDPD.Ntaps = Ntaps
        
        DPD = ETDNN(paramDPD.layers, activation = nn.ReLU() ).to(paramDPD.device)
        DPD.load_state_dict(th.load(study_path + rf"\models\ETDNN_model_trial{trial}.pth", weights_only = True))
        
        
    elif paramDPD.model == "ETDKAN":        
        N1      = np.loadtxt(study_path + r"\parameters\N1.txt").astype(np.int64)[trial]
        Ntaps   = np.loadtxt(study_path + r"\parameters\Ntaps.txt", dtype = int).astype(np.int64)[trial]
        k       = np.loadtxt(study_path + r"\parameters\k.txt", dtype = int).astype(np.int64)[trial]
        grid    = np.loadtxt(study_path + r"\parameters\grid.txt", dtype = int).astype(np.int64)[trial]
        
        paramDPD.Ntaps  = Ntaps
        paramDPD.layers = [Ntaps, N1, Ntaps]
        paramDPD.k      = k
        paramDPD.grid   = grid
        
        DPD = ETDKAN(paramDPD.layers, grid, k, paramDPD.seed, paramDPD.device)    
        DPD.KAN = DPD.KAN.loadckpt(study_path + rf"\models\ETDKAN_model_trial{trial}")
        
    elif paramDPD.model == "MP":    
        P = np.loadtxt(study_path + r"\parameters\P.txt").astype(np.int64)[trial]
        M = np.loadtxt(study_path + r"\parameters\M.txt").astype(np.int64)[trial]
        
        paramDPD.P = P
        paramDPD.M = M
        
        DPD = np.loadtxt(study_path + rf"\models\MP_model_trial{trial}.txt", dtype = np.complex128)
            
    else:
        print("DPD model not in the list")
    
    # Test as DPD
    paramDPD.DPD = DPD
    sigRx_PA_DPD, symbRx_DPD = test_as_dpd(DPD, symbTx, paramRoF, paramDPD)
    
    # EVM, ACLR calculation
    discard = 500
    index = np.arange(discard, symbRx_DPD.size - discard)
    paramOFDM = paramRoF.paramOFDM
    
    freq, P_sigRx_PA_DPD = welch(pnorm(sigRx_PA_DPD)[0::paramOFDM.SpS//SpS_DPD], fs = SpS_DPD*paramOFDM.Rs, nfft = 16*1024, return_onesided = False)
    
    ACLR   = calcACLR(P_sigRx_PA_DPD, freq, paramOFDM.bw/2, 2.5e6)
    EVM    = np.sqrt(calcEVM(symbRx_DPD[index], paramOFDM.modOrder, paramOFDM.modType)[0])*100
    NFLOPs = calcNFLOPs(paramDPD)
        
    return EVM, ACLR, NFLOPs


#if paramDPD.symb:
#    DPD_model.set_symb()
#    y_symb, y_func = DPD_model.get_symb()
#    y_symb_str = [str(y_symb[i]) for i in range(len(y_symb)) ]
    
#    file_name = model_path + f"\\EKAN_model_trial{trial.number+1}.txt"

#    with open(file_name, "w") as f:
#        for func in y_symb_str:
#            f.write(func+"\n"+"\n")
#        f.close()
