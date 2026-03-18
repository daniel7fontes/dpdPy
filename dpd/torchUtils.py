import contextlib
import numpy as np
import torch as th

from torch.utils.data import Dataset
from numba import njit
from tqdm  import tqdm


class slidingWindowDataSet(Dataset):
    """
    A custom dataset class for creating sliding window samples from a signal.

    Args:
        x (numpy.ndarray): Input signal.
        y (numpy.ndarray): Array of corresponding targets.
        Ntaps (int): Number of taps/window size.
        SpS (int, optional): Samples per symbol. Defaults to 1.

    Attributes:
        Ntaps (int): Number of taps/window size.
        SpS (int): Samples per symbol.
        x (numpy.ndarray): Input signal padded with zeros.
        y (numpy.ndarray): Array of corresponding targets.

    Methods:
        __getitem__(self, idx): Retrieves the item at the specified index.
        __len__(self): Returns the total number of items in the dataset.
    """

    def __init__(self, x, y, Ntaps, K, SpS=1, c=False, augment = False):
        """
        Initialize the slidingWindowDataSet.

        Args:
            x (numpy.ndarray): Input signal.
            y (numpy.ndarray): Array of corresponding targets.
            Ntaps (int): Number of taps/window size.
            SpS (int, optional): Samples per symbol. Defaults to 1.
        """
        super(slidingWindowDataSet, self).__init__()
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

        Args:
            idx (int): Index of the item.

        Returns:
            tuple: A tuple containing the input and target tensors.
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

        Returns:
            int: Total number of items in the dataset.
        """
        return (len(self.x) - self.Ntaps) // self.SpS


def augmentFeatures(x, K):
    """
    Augment the features of a complex-valued tensor.

    Parameters
    ----------
    x : torch.Tensor
        Input tensor with complex values.

    Returns
    -------
    torch.Tensor
        Tensor with augmented features. Each column contains:
        - Real part of the input tensor.
        - Imaginary part of the input tensor.
        - Absolute value of the input tensor.
        - Absolute squared value of the input tensor.
        - Absolute value to the third power of the input tensor.
    """
    
    x_list = [x.real, x.imag]
    
    for k in range(K):
        x_list.append(th.abs(x)**(k+1))

    return th.stack(x_list, dim = 1)
    

def fitFilterNN(x, model, paramTrain, paramModel, batchSize = 100, predict = True):
    
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


def batchData(data_input, data_label, batchSize, shuffle = False):
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
    
    train_dataset = slidingWindowDataSet(sigIn[indIn_Train], sigRef[indRef_Train], M + 1, K, augment = augment)
    test_dataset  = slidingWindowDataSet(sigIn[indIn_Test], sigRef[indRef_Test], M + 1, K, augment = augment)
    
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



@njit
def MP_filter(x, w, M, P):
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


@njit
def slidingWindowMP(x, i, M, P):
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