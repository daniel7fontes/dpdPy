"""
================================================================
Model training utilities (:mod:`dpd.train`)
================================================================

   slidingWindowMP  -- Create sliding window for MP input model.
   slidingWindowNN  -- Create sliding window for NN input models.
   augmentFeatures  -- Return augmented tensor.
   batchData        -- Separate data into batches.
   createDatasets   -- Create datasets for NN models training.
   trainMP          -- Train MP model.
   trainNN          -- Train NN models.
   
"""

"""Model training utilities."""


import numpy as np
import torch as th

from torch            import nn
from torch.utils.data import Dataset
from numba            import njit
from tqdm.notebook    import tqdm
from dpd.models       import ARVTDNN, ETDNN, ETDKAN, MP


@njit
def slidingWindowMP(x, i, M, P):
    """
    Create the sliding window for MP in the form x[i] = [x[i] x[i-1] ... x[i]|x[i]| x[i-1]|x[i-1]| ... x[i-M]|x[i-M]|**(P-1) ]

    Parameters
    ----------
    x : np.array
        Input complex-valued signal to MP model
    
    i : int
        Index for sliding window    
    
    M : int 
        Memory length of the model

    P : int 
        Maximum power order of the model
        
    Returns
    -------
    x_win : np.array
        Sliding window in MP format
    
    """
    ind = np.arange(0, M + 1)
    dataSize = x.size
    
    x_extend = np.zeros(2*M + dataSize, dtype = np.complex128)
    for n in range(dataSize):
        x_extend[n] = x[n]
    
    x_win = np.zeros(P * (M + 1), dtype = np.complex128)
    X = x_extend[i - ind]
    j = 0
    
    for p in range(P):
        for m in range(M + 1):
            x_win[j] = X[m] * (np.abs(X[m]) ** p)
            j += 1

    return x_win


class slidingWindowNN(Dataset):
    def __init__(self, x, y, Ntaps, K, SpS = 1, augment = False):
        """
        Initialize a custom dataset class for creating sliding window samples from a signal.

        x : th.tensor 
            Input signal.

        y : int 
            Array of corresponding targets.
        
        Ntaps : int
            Number of taps/window size.
        
        K : int
            Max power of the input amplitude (for ARVTDNN)
            
        SpS : int
            Samples per symbol. Default is 1.
        
        augment : boolt
            Flag to indicate if the input contains augmented terms. Default is False
        
        """
        
        super(slidingWindowNN, self).__init__()
        self.Ntaps = Ntaps
        self.SpS = SpS
        
        x_pad = th.nn.functional.pad(x, (Ntaps // 2, Ntaps // 2), "constant", 0)
        
        if augment:
            self.x = augmentFeatures(x_pad, K).to(th.float32)
        else:
            self.x = th.view_as_real(x_pad).to(th.float32)

        self.y = th.view_as_real(y).to(th.float32)
        self.augment = augment

    def __getitem__(self, idx):
        """
        Retrieves the item at the specified index.
        
        """

        center_idx = idx * self.SpS + self.Ntaps // 2
        start_idx = center_idx - self.Ntaps // 2
        end_idx = center_idx + self.Ntaps // 2

        if start_idx == end_idx:
            inputs = self.x[center_idx, :].flatten()
        else:
            inputs = self.x[start_idx:end_idx, :].flatten()

        target = self.y[idx, :].flatten()

        return inputs, target

    def __len__(self):
        """
        Returns the total number of items in the dataset.

        """
        return (len(self.x) - self.Ntaps) // self.SpS


def augmentFeatures(x, K):
    """
    Return a tensor with real and imaginary parts of x, and its amplitude from 1 to Kth power

    Parameters
    ----------
    x : th.tensor
        Input complex-valued tensor
    
    K : int
        Maximum power for amplitude of x
    
    """    
    
    x_list = [x.real, x.imag]
    
    for k in range(K):
        x_list.append(th.abs(x)**(k+1))

    return th.stack(x_list, dim = 1)


def batchData(data_input, data_label, batchSize, shuffle = False):
    """
    Divide a dataset into batches

    Parameters
    ----------
    data_input : th.tensor
        Input data to neural network model
    
    data_label : th.tensor
        Reference data to neural network model
    
    batchSize : int
        Size of the batches
    
    shuffle : bool
        Flag to indicate whether to shuffle the data
    
    Returns
    -------
    data_dic : dictionary
        Dictionary with the input and label data divided in batches
    
    """
    
    dataSize   = data_input.shape[0]  # mudar nome desse param
    numBatches = int(np.floor(dataSize / batchSize))

    data_dic = {}
    
    for b in range(numBatches):
        index = np.arange(b*batchSize, (b+1)*batchSize, dtype = int)
        
        if shuffle:
            np.random.shuffle(index)
        
        data_dic[f"batch_{b}"] = {"input" : data_input[index,:], 
                                  "label" : data_label[index,:]}
    if numBatches*batchSize < dataSize:
        index = np.arange((b+1)*batchSize, dataSize, dtype = int)
        
        if shuffle:
            np.random.shuffle(index)
        
        data_dic[f"batch_{b+1}"] = {"input" : data_input[index,:], 
                                    "label" : data_label[index,:]}
        
    return data_dic


def createDatasets(sigIn, sigRef, paramTrain, paramModel):
    """
    Create the labeled datasets divided in batches for NN DPD models training

    Parameters
    ----------
    sigIn : np.array
        Complex signal at the input of the model
    
    sigRef : np.array
        Complex signal for reference
    
    paramTrain : optic.utils.parameters object
        An object containing the parameters for model training.
        - paramTrain.trainTestFrac : float
            Fraction of the data used for training
            
        - paramTrain.batchSize
            Size of the batches for training
            
        - paramTrain.shuffle
            Flag to indicate whether to shuffle the training/test data
    
    paramModel : optic.utils.parameters object
        An object containing the specification for model hyperparameters.
        - paramModel.M : int 
            Memory length of the model

        - paramModel.K : int 
            Maximum power order of the model (for ARVTDNN)

    Returns
    -------
    train_dataloader : dict
        Dictionary containing the labeled data divided in batches for training
    
    test_dataloader : dict
        Dictionary containing the labeled data divided in batches for test
        
    """    
    
    
    trainTestFrac = paramTrain.trainTestFrac
    batchSize     = paramTrain.batchSize
    shuffle       = paramTrain.shuffle
    device        = paramTrain.device
    shuffle       = paramTrain.shuffle
    
    model_name = paramModel.model_name
    M = paramModel.M 
    
    if model_name == "ARVTDNN":
        K = paramModel.K
        augment = True
    else:
        K = 0
        augment = False
    
    # Create the datasets
    indIn_Train = th.arange(0, int(trainTestFrac * len(sigIn)))
    indIn_Test  = th.arange(int(trainTestFrac * len(sigIn)), len(sigIn))
    
    indRef_Train = th.arange(0, int(trainTestFrac * len(sigRef)))
    indRef_Test  = th.arange(int(trainTestFrac * len(sigRef)), len(sigRef))
    
    train_dataset = slidingWindowNN(sigIn[indIn_Train], sigRef[indRef_Train], M + 1, K, augment = augment)
    test_dataset  = slidingWindowNN(sigIn[indIn_Test], sigRef[indRef_Test], M + 1, K, augment = augment)
    
    # Train dataloader
    train_inputs = th.empty((0, (2 + K)*(M + 1)), device = device) if model_name == "ARVTDNN" else th.empty((0, 2*(M + 1)), device = device)
    train_labels = th.empty((0, 2), device = device)
    
    for data, label in train_dataset:
        train_inputs = th.cat((train_inputs, data.reshape(1, -1).to(device)), dim = 0)
        train_labels = th.cat((train_labels, label.reshape(1, -1).to(device)), dim = 0)
    
    batch_train = batchSize if batchSize <= train_inputs.shape[0] else train_inputs.shape[0]
    train_dataloader = batchData(train_inputs, train_labels, batch_train, shuffle)
    
    # Test dataloader
    test_inputs = th.empty((0, (2 + K)*(M + 1)), device = device) if model_name == "ARVTDNN" else th.empty((0, 2*(M + 1)), device = device)
    test_labels = th.empty((0, 2), device = device)
     
    for data, label in test_dataset:
        test_inputs = th.cat((test_inputs, data.reshape(1, -1).to(device)), dim = 0)
        test_labels = th.cat((test_labels, label.reshape(1, -1).to(device)), dim = 0)
    
    batch_test = batchSize if batchSize <= test_inputs.shape[0] else test_inputs.shape[0]
    test_dataloader = batchData(test_inputs, test_labels, batch_test, shuffle)
    
    return train_dataloader, test_dataloader


def trainMP(sigIn, sigRef, paramTrain, paramModel):
    """
    Train a memory polynomial to find its parameters.

    Parameters
    ----------
    sigIn : np.array
        Complex signal at the input of the MP model
    
    sigRef : np.array
        Complex signal for reference
    
    paramTrain : optic.utils.parameters object
        An object containing the parameters for MP training.
        
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
            
        - paramTrain.pgrsBar : bool
            Flag to indicate whether a progress bar is shown.
    
    paramModel : optic.utils.parameters object
        An object containing the specification for MP hyperparameters.
        - paramModel.M : int 
            Memory length of the model

        - paramModel.P : int 
            Maximum power order of the model

    Returns
    -------
    model : object
        Object of the class MP(M, P) with optimized weights
    
    trainLoss : np.array
        Real-valued array with MSE by epoch of the training stage

    """    
    
    M = paramModel.M
    P = paramModel.P
        
    mu      = paramTrain.mu
    lbd     = paramTrain.lbd
    S       = paramTrain.S
    alg     = paramTrain.alg
    epochs  = paramTrain.epochs
    pgrsBar = paramTrain.pgrsBar
    
    model   = MP(M, P)
    
    dataSize  = sigIn.size
    trainLoss = np.zeros(epochs)
    
    for t in tqdm(range(epochs), disable = not(pgrsBar) ):
        err = np.zeros(dataSize, dtype = complex)
        
        for i in range(dataSize):
        
            y_win  = slidingWindowMP(sigIn, i, M, P)
            y      = model.forward(y_win)[0]
            
            err[i] = sigRef[i] - y
                        
            if alg == "RLS":
                g = (1/lbd) * (S @ y_win.reshape(P*(M+1), 1) ) / ( 1 + (1/lbd)* np.conj(y_win.reshape(1, P*(M+1))) @ S @ y_win.reshape(P*(M+1), 1) )
                S = (1/lbd) * S - (1/lbd)*g.reshape(P*(M+1), 1) @ np.conj(y_win.reshape(1, P*(M+1))) @ S
                model.w += g * np.conj(err[i])
                
            elif alg == "LMS":
                model.w += mu * np.conj(err[i]) * y_win.reshape((P*(M+1), 1))
                
            else:
                print("No alg...") # add error exception
            
        trainLoss[t] = np.mean(np.abs(err)**2)
    
    return model, trainLoss


def trainNN(sigIn, sigRef, paramTrain, paramModel):
    """
    Train a neural network-based model (ARVTDNN, ETDNN, ETDKAN) to find its parameters.

    Parameters
    ----------
    sigIn : np.array
        Complex signal at the input of the NN model
    
    sigRef : np.array
        Complex signal for reference
    
    paramTrain : optic.utils.parameters object
        An object containing the parameters for NN training.
        
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

    paramModel : optic.utils.parameters object
        An object containing the specification for NN hyperparameters.
        - paramModel.M : int 
            Memory length of the model

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
            Seed for ETDKAN parameters initialization

    Returns
    -------
    model : object
        Neural network model
    
    trainLoss : np.array
        Real-valued array with MSE by epoch of the training stage

    testLoss : np.array
        Real-valued array with MSE by epoch of the test stage
    """    
    
    adaptLearningRatio = paramTrain.adaptLearningRatio
    lr      = paramTrain.lr
    epochs  = paramTrain.epochs
    
    pgrsBar = paramTrain.pgrsBar
    device  = paramTrain.device
    
    model_name = paramModel.model_name
    
    activation_dict = {'leaky_relu': nn.LeakyReLU(), 'relu': nn.ReLU(), 'sigmoid': nn.Sigmoid(), 'tanh': nn.Tanh(), 'linear': nn.Identity()} 
    
    if model_name == "ARVTDNN":
        M = paramModel.M 
        K = paramModel.K 
        hidden_layers = paramModel.hidden_layers
        activation = paramModel.activation
        
        model = ARVTDNN(M, K, hidden_layers, activation_dict[activation]).to(device)
        
    elif model_name == "ETDNN":
        M = paramModel.M 
        N = paramModel.N
        activation = paramModel.activation
        
        model = ETDNN(M, N, activation_dict[activation]).to(device)
        
    elif model_name == "ETDKAN":
        M = paramModel.M 
        N = paramModel.N
        k = paramModel.k
        grid = paramModel.grid
        seed = paramModel.seed
        
        model = ETDKAN(M, N, k, grid, seed, device).to(device)
                
    else:
        print("No model")
    
    # Loss and optimizer definition
    loss_fn   = nn.MSELoss()
    optimizer = th.optim.Adam(model.parameters(), lr = lr)
    
    trainLoss = np.zeros(epochs)
    testLoss  = np.zeros(epochs)
    
    # Prepare dataset for training and test
    train_dataloader, test_dataloader = createDatasets(sigIn, sigRef, paramTrain, paramModel)
    numBatches_train = len(train_dataloader)
    numBatches_test  = len(test_dataloader)
    
    for t in tqdm(range(epochs), disable = not(pgrsBar)):
        # Training
        model.train()
        trainLoss[t] = 0
        
        for batch, (_, batch_data) in enumerate(train_dataloader.items()):
            X = batch_data['input']
            y = batch_data['label']
        
            # Compute prediction error
            pred = model(X)
            loss = loss_fn(pred, y)
            
            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            trainLoss[t] += loss.item()
        
        trainLoss[t] /= numBatches_train
        
        # Validation
        model.eval()
        testLoss[t] = 0
        
        with th.no_grad():
            for batch, (_, batch_data) in enumerate(test_dataloader.items()):
                X = batch_data['input']
                y = batch_data['label']
                
                pred = model(X)
                loss = loss_fn(pred, y)
                
                testLoss[t] += loss.item()
            
        testLoss[t] /= numBatches_test
        
        if adaptLearningRatio:
            if (t + 1) % 100 == 0:
                for g in optimizer.param_groups:
                    g['lr'] = g['lr']/2
                    
        if model_name == "ETDKAN":
            symbolicEpoch = paramTrain.symbolicEpoch
            
            if t == symbolicEpoch - 1:
                model.set_symb()
                
                for g in optimizer.param_groups:
                    g['lr'] = 1e-5
        
    return model, trainLoss, testLoss