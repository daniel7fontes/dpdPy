"""
================================================================
Utilities for Optuna hyperparameter optimization of a DPD model for ARoF (:mod:`dpd.optuna_tasks`)
================================================================

   get_pareto         -- Get the Pareto front solutions of a optimization with two objective functions.
   get_best_pareto    -- Get the best solution from Pareto front by the criterion of minimum distance to the ideal solution.
   objective_rof_dpd  -- For an Optuna trial, train and test a DPD model, with the corresponding hyperparameter set of the trial, in an ARoF link.
   
"""

"""Utilities for Optuna hyperparameter optimization of a DPD model for ARoF."""

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
    """
    Get the Pareto front solutions of a optimization with two objective functions    

    Parameters
    ----------
    f1 : np.array
        Output of the objective function f1 for each trial
    
    f2 : np.array
        Output of the objective function f2 for each trial
    
    n_trials : int
        Number of optization trials
    
    Returns
    -------
    pareto_solutions : np.array
        Pairs of f1 and f2 values from the Pareto front
        
    pareto_trials : np.array
        Corresponding trials of the Pareto front solutions
    
    """
    
    solutions = np.hstack( (f1.reshape((n_trials, 1)), f2.reshape((n_trials, 1))) )
    pareto = []
    
    for i, s in enumerate(solutions):
        test = np.where( (s[0] > solutions[:,0]) * (s[1] > solutions[:,1]) )[0]
        
        if (not(len(test)) and not(np.isnan(s[0])) and not(np.isnan(s[1])) ):
            s_pareto = np.append(s, i+1)
            pareto.append(tuple(s_pareto))

    pareto.sort()
    pareto = np.array(pareto)

    pareto_trials = pareto[:,2].astype(np.int64)
    pareto_solutions = pareto[:,0:2]
    
    return pareto_solutions, pareto_trials    


def get_best_pareto(pareto_solutions, weights = (1, 1)):
    """
    Get the best solution from Pareto front by the criterion of minimum distance to the ideal solution
    
    Parameters
    ----------
    pareto_solutions : np.array
        Pairs of f1 and f2 (objective functions) values from the Pareto front
    
    weights : tuple
        Importance weight relative to each objective function for distance calculation 
        
    Returns
    -------
    best : np.array
        Pair of f1 and f2 closer to the ideal solution point
    
    best_arg : int
        Index of the pareto_solutions that contains the best solution    
    
    ideal : np.array
        Ideal solution (min(f1), min(f2))
    
    pareto_trials : np.array
        Corresponding trials of the Pareto front solutions
    
    """
    f1 = pareto_solutions[:,1]
    f2 = pareto_solutions[:,0]
    w1, w2 = weights
    
    num_sol = f1.size

    gamma_1 = 1/(w1*np.abs(np.max(f1)))
    gamma_2 = 1/(w2*np.abs(np.max(f2)))
    
    ideal = np.array([np.min(f1), np.min(f2)])
    
    distance_pareto = np.zeros(num_sol)
    for i in range(num_sol):
        distance_pareto[i] = np.sqrt( (gamma_1*(f1[i] - ideal[0]))**2 + (gamma_2*(f2[i] - ideal[1]))**2 )

    best_arg = np.argmin(distance_pareto)
    best = (f1[best_arg], f2[best_arg])
        
    return best, best_arg, ideal
    

def objective_rof_dpd(trial, data, paramOFDM, paramRoF, paramModel, paramTrain, paramMetrics):
    """
    For an Optuna trial, train and test a DPD model, with the corresponding hyperparameter set of the trial, in an ARoF link
    
    Parameters
    ----------
    
    trial : optuna trial object
    
    data : optic.utils.parameters object
        Object containing the data specifications for models training
        
        - data.sigIn : np.array 
            Complex signal at the input of the model (at DPD sampling frequency)

        - data.sigRef : np.array
            Complex signal for reference (at DPD sampling frequency)
        
        - data.sigTx : np.array
            Complex signal for reference
        
        - data.Rs : float
            Symbol rate of the transmitted signal (Symbols/s)
        
        - data.SpS : int
            Samples per symbol of the transmitted signal
        
        - data.Fs : float
            Sampling frequency of the transmitted signal (Samples/s)
            
        - data.Fs_DPD : float
            DPD sampling frequency of the transmitted signal at (Samples/s)
        
        - data.modOrder : int
            Number of constellation symbols
        
        - data.constType : string
            Constellation type
        
    paramOFDM : optic.utils.parameters object
        Parameters for OFDM modulation.

        - param.Nfft : int
            Size of the FFT.
            
        - param.G : int
            Cyclic prefix length
            
        - param.hermitSymmetry : bool
            If True, indicates real OFDM symbols; if False, indicates complex OFDM symbols
            
        - param.pilot : complex-valued float
            Pilot symbol
        
        - param.pilotCarriers : np.array
            Indexes of pilot subcarriers. [default: empty array].
        
        - param.nullCarriers : np.array
            Indexes of null subcarriers
            
        - param.SpS : int
            Samples per symbol of the transmitted signal
    
    
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


    paramModel : optic.utils.parameters object
        An object containing the specification for DPD hyperparameters.
        (for MP)
        - paramModel.M : int 
            Memory length of the model

        - paramModel.P : int 
            Maximum power order of the model
        
        (for ARVTDNN)
        K : int 
            Maximum power order of the model
        
        hidden_layers : list
            Number of layers for each hidden layer (for ARVTDNN)
        
        activation : string
            Activation function for hidden layers (for ARVTDNN: "leaky_relu", "relu", "sigmoid", "tanh", "linear")
        
        (for ETDNN)
        N : int 
            Size of the hidden layer
            
        (for ETDKAN)
        k : int
            B-spline polynomials order
            
        grid : int 
            B-spline grid
            
        seed : int
            Seed for ETDKAN parameters initialization
        
        device : string
            Processing device name ("cpu" or "cuda")
        
    
    paramTrain : optic.utils.parameters object
        An object containing the parameters for MP training.
        
        (for MP)
        - paramTrain.alg : string
            Adaptive algorithm name ("RLS" or "LMS")
        
        - paramTrain.epochs : int 
            Number of training epochs.

        - paramTrain.mu : float 
            Learning rate (for LMS).
        
        - paramTrain.lbd : float
            Forgetting factor (for RLS)
        
        - paramTrain.S : np.array
            Initial inverse correlation matrix for RLS algorithm
        
        (for NN models)
        - paramTrain.lr : float
            Learning rate
        
        - paramTrain.epochs : int 
            Number of training epochs.

        - paramTrain.adaptLearningRatio : bool 
            Flag that indicates whether the learning rate drops by half every 100 epochs.
        
        - paramTrain.device : string
            Processing device name ("cpu" or "cuda")
        
        - paramTrain.pgrsBar : bool
            Flag to indicate whether a progress bar is shown
            
        - paramTrain.trainTestFrac : float
            Fraction of the data used for training
            
        - paramTrain.batchSize
            Size of the batches for training
            
        - paramTrain.shuffle
            Flag to indicate whether to shuffle the training/test data
        
        - paramTrain.pgrsBar : bool
            Flag to indicate whether a progress bar is shown.
            
    paramMetrics : optic.utils.parameters object
        An object containing the parameters for MP training.
        
        - paramMetrics.bw_for_aclr : float
            Bandwidth limit for ACLR calculation (Hz)
        
        - paramMetrics.offset_for_aclr : float
            Frequency offset in bandwidth limit for ACLR calculation (Hz)
        
        - paramMetrics.discard : int
            Number of symbols to discard for metrics calculation
                
        - paramMetrics.metrics : list
            List of strings containing the metrics to be calculated as objective functions (["EVM", "ACLR", "NFLOP"])
    
    
    Returns
    -------
    out : tuple
        Metrics calculated for the trial

    """
    
    
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
        
    elif model_name == "ETDNN":
        paramModel.M = trial.suggest_int("M", 1, 19, step = 2)
        paramModel.N = trial.suggest_int("N", 1, 20)
        
        model, trainLoss, testLoss = trainNN(sigIn, sigRef, paramTrain, paramModel)
        
    elif model_name == "ETDKAN":
        paramModel.M    = trial.suggest_int("M", 1, 3, step = 2)
        paramModel.N    = trial.suggest_int("N", 1, 4) 
        paramModel.k    = trial.suggest_int("k", 2, 6)
        paramModel.grid = trial.suggest_int("grid", 2, 6)
        
        model, trainLoss, testLoss = trainNN(sigIn, sigRef, paramTrain, paramModel)        
    
    elif model_name == "MP":    
        paramModel.P = trial.suggest_int("P", 1, 10)
        paramModel.M = trial.suggest_int("M", 0, 10)
        paramTrain.S = 5e-2*np.eye(paramModel.P*(paramModel.M + 1), dtype = complex)
        
        model, trainLoss = trainMP(sigIn, sigRef, paramTrain, paramModel)
        
    else:
        print("DPD model not in the list")
        
    sigTx_DPD, gain_DPD = applyDPD(sigTx, model, Rs, Fs, Fs_DPD, paramModel)
        
    if np.isnan(gain_DPD):
        EVM  = np.nan
        ACLR = np.nan
        NFLOP = np.nan
    
    else:
        sigRx_PA_DPD = RoF_channel(sigTx_DPD, paramRoF, gain_DPD, filter_numtaps = 4096)
        
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
        
        # Saving models
        file_path = model_path + f"\\{model_name}_trial_{trial.number + 1}"
        
        if model_name == "MP":
            file_path += ".txt"
        elif model_name == "ETDNN" or "ARVTDNN":
            file_path += ".pth"
        
        model.save(file_path)
        
        # Saving train/test losses
        np.savetxt(model_path + f"\\{model_name}_trainLoss_trial_{trial.number + 1}.txt", trainLoss, fmt = "%f")
        if model_name != "MP":
            np.savetxt(model_path + f"\\{model_name}_testLoss_trial_{trial.number + 1}.txt", testLoss, fmt = "%f")
        
        # Saving metrics
        np.savetxt(model_path + rf"\\{model_name}_metrics_{trial.number + 1}.txt", full_metrics_array, fmt = '%f')

    # Calculate only the metrics for Optuna trial output
    metrics_dic = {"EVM" : EVM, "ACLR" : ACLR, "NFLOP" : np.mean(np.array(NFLOP))}
    out = []
    
    for m in metrics_dic.keys():
        if m in metrics:
            out.append(metrics_dic[m])
    
    out = tuple(out)
        
    return out