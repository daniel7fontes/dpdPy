"""
================================================================
Classes for each DPD model (:mod:`dpd.models`)
================================================================

   MP       -- A class implementing the functionalities of the MP model.
   ARVTDNN  -- A class implementing the functionalities of the ARVTDNN model.
   ETDNN    -- A class implementing the functionalities of the ETDNN model.
   ETDKAN   -- A class implementing the functionalities of the ETDKAN model.
   
"""

"""Classes for DPD models."""


import numpy as np
import torch as th
import kan   as kn

from torch import nn


class MP:
    def __init__(self, M, P):
        """
        Initialize the MP model. The MP coefficients are represented in an array w of size (M+1)*N initialized in the form [1, 0, ..., 0]

        M : int 
            Memory length of the model

        P : int 
            Maximum power order of the model
            
        References
        ----------
        [1] D. R. Morgan et al., "A generalized memory polynomial model for digital predistortion of RF power amplifiers", IEEE Transactions on Signal Processing, vol. 54, n. 10, p. 3852-3860, 2006, doi: 10.1109/TSP.2006.879264.
        
        [2] F. Ghannouchi, O. Hammi and M. Helaoiu, "Behavioral modeling and predistortion of wideband wireless transmitters", Nashville, Wiley, 2015, ISBN 9781119004424.
        
        """
        
        self.M = M
        self.P = P
        self.w = np.zeros((self.P * (self.M + 1), 1), dtype = complex)
        self.w[0] = 1
        
    def forward(self, x):
        """
        Calculate the output of the model
        
        Parameters
        ----------
        x : np.array
            Sliding window in MP format
            
        Returns
        -------
        Dot product between MP coefficients and sliding window
        
        """

        return np.dot(x, np.conj(self.w))
    
    def get_param(self):
        """
        Return the model attributes (M, P)
        
        """

        return (self.M, self.P)
    
    def calcNFLOP(self):
        """
        Calculate the number of floating point operations (NFLOP) for a forward step of the model
        
        Returns
        -------
        NFLOP of the model
        
        """
        
        return (self.M + 1) * (11*self.P + 6) - 1
        
    def save(self, model_path):
        """
        Save the model coefficients in a .txt file
        
        Parameters
        ----------
        model_path : string
            Path where the file will be stored
        
        """
        
        np.savetxt(model_path, self.w, fmt = '%f')
        
    def load(self, model_path):
        """
        Load a model coefficients from a .txt file
        
        Parameters
        ----------
        model_path : string
            Path where the loaded file is stored
        
        """
        
        self.w = np.loadtxt(model_path, dtype = np.complex128)
        

class ARVTDNN(nn.Module):
    def __init__(self, M, K, hidden_layers, activation):
        """
        Initialize the ARVTDNN model. 
        
        M : int 
            Memory length of the model

        K : int 
            Maximum power order of the model
        
        hidden_layers : list
            Number of layers for each hidden layer (for ARVTDNN)
        
        activation : string
            Activation function for hidden layers (for ARVTDNN: "leaky_relu", "relu", "sigmoid", "tanh" or "linear")
        
        References
        ----------
        [1] D. Wang et al., "Augmented real-valued time-delay neural network for compensation of distortions and impairments in wireless transmitters", IEEE Transactions on Neural Networks and Learning Systems, vol. 30, n. 1, p. 242-254, 2019, doi: 10.1109/TNNLS.2018.2838039.

        """
        
        super(ARVTDNN, self).__init__()
        self.layers = nn.ModuleList()
        self.activation = activation
        
        self.M = M
        self.K = K
        self.layers_size = []
        
        self.layers_size.append((self.K + 2)*(self.M + 1))
        self.layers_size.append(hidden_layers[0])
        
        self.layers.append(nn.Linear((self.K + 2)*(self.M + 1), hidden_layers[0] ))
        
        for i in range(0, len(hidden_layers) - 1):
            self.layers.append(nn.Linear(hidden_layers[i], hidden_layers[i + 1]))
            self.layers_size.append(hidden_layers[i+1])
        
        self.layers.append(nn.Linear(hidden_layers[-1], 2))
        self.layers_size.append(2)

    def forward(self, x):
        """
        Calculate the output of the model
        
        Parameters
        ----------
        x : th.tensor
            Sliding window in NN format
            
        Returns
        -------
        Output of the ARVTDNN with x as input
        
        """
        
        for ind, layer in enumerate(self.layers):
            x = self.activation(layer(x)) if ind < len(self.layers_size) - 2 else layer(x)
        return x
    
    def get_param(self):
        """
        Return the model attributes (M, K, size of the layers)
        
        """
        
        return (self.M, self.K, self.layers_size)
    
    def calcNFLOP(self):
        """
        Calculate the number of floating point operations (NFLOP) for a forward step of the model
        
        Returns
        -------
        NFLOPs : int
            NFLOP of the model
        
        """
        
        NFLOPs = 0
        
        for i in range(len(self.layers_size) - 1):
            NFLOPs += 2*self.layers_size[i] * self.layers_size[i+1] 
            
            if i < (len(self.layers_size) - 2):
                NFLOPs += self.layers_size[i+1]
        
        NFLOPs += 10*(self.M + 1) + (self.K - 1) * (self.M + 1)
        
        return NFLOPs
    
    def save(self, model_path):
        """
        Save the model coefficients in a .pth file
        
        Parameters
        ----------
        model_path : string
            Path where the file will be stored
        
        """
        
        th.save(self.state_dict(), model_path)
        
    def load(self, model_path):
        """
        Load a model coefficients from a .pth file
        
        Parameters
        ----------
        model_path : string
            Path where the loaded file is stored
        
        """
        
        self.load_state_dict(th.load(model_path, weights_only = True))


class ETDNN(nn.Module):
    def __init__(self, M, N, activation):
        """
        Initialize the ETDNN model. 
        
        M : int 
            Memory length of the model

        N : int 
            Size of the hidden layer
        
        References
        ----------
        [1] M. Tanio, N. Ishii and N. Kamiya, "Efficient digital predistortion using sparse neural network", IEEE Access, vol. 8, p. 117841-117852, 2020, doi: 10.1109/ACCESS.2020.3005146.
        
        """
        
        super(ETDNN, self).__init__()
        self.layers = nn.ModuleList()
        self.activation = activation
        
        self.M = M
        self.N = N 
        
        self.layer1 = nn.Linear(self.M + 1, self.N)
        self.layer2_real = nn.Linear(self.N, self.M + 1)
        self.layer2_imag = nn.Linear(self.N, self.M + 1)
        

    def forward(self, x):
        """
        Calculate the output of the model
        
        Parameters
        ----------
        x : th.tensor
            Sliding window in NN format
            
        Returns
        -------
        Output of the ETDNN with x as input
        
        """
        
        batch_size = x.shape[0]
        
        x_complex = th.view_as_complex(x.reshape((batch_size, self.M + 1, 2)))
        x_abs     = th.abs(x_complex)
        
        # MLP
        x_out = self.activation(self.layer1(x_abs))
        x_out = self.layer2_real(x_out) + 1j*self.layer2_imag(x_out)
        
        y = th.sum(x_out*x_complex, axis = 1)
        y = th.view_as_real(y)
            
        return y
    
    def get_param(self):
        """
        Return the model attributes (M, N)
        
        """
        
        return (self.M, self.N)
    
    def calcNFLOP(self):
        """
        Calculate the number of floating point operations (NFLOP) for a forward step of the model
        
        Returns
        -------
        NFLOP of the model
        
        """
        
        return (self.M + 1)*(6*self.N + 18) + self.N - 2
    
    def save(self, model_path):
        """
        Save the model coefficients in a .pth file
        
        Parameters
        ----------
        model_path : string
            Path where the file will be stored
        
        """
        
        th.save(self.state_dict(), model_path)

    def load(self, model_path):
        """
        Load a model coefficients from a .pth file
        
        Parameters
        ----------
        model_path : string
            Path where the loaded file is stored
        
        """
        
        self.load_state_dict(th.load(model_path, weights_only = True))


class ETDKAN(nn.Module):
    def __init__(self, M, N, k, grid, seed, device):
        """
        Initialize the ETDKAN model. 
        
        M : int 
            Memory length of the model

        N : int 
            Size of the hidden layer
            
        k : int
            B-spline polynomials order
            
        grid : int 
            B-spline grid
            
        seed : int
            Seed for ETDKAN parameters initialization
        
        device : string
            Processing device name ("cpu" or "cuda")
        
        References
        ----------
        [1] Z. Liu et al., "KAN: Kolmogorov-Arnold Networks", The Thirteenth International Conference on Learning Representations, 2025, doi: https://doi.org/10.48550/arXiv.2404.19756
        
        [2] M. Tanio, N. Ishii and N. Kamiya, "Efficient digital predistortion using sparse neural network", IEEE Access, vol. 8, p. 117841-117852, 2020, doi: 10.1109/ACCESS.2020.3005146.
        
        """
        
        super(ETDKAN, self).__init__()
        self.grid   = grid
        self.k      = k
        self.seed   = seed
        self.device = device
        
        self.M = M
        self.N = N 
        
        self.symb = False
        self.KAN = kn.KAN(width = [self.M + 1, self.N, 2*(self.M + 1)], grid = self.grid, k = self.k, seed = self.seed, device = self.device, auto_save = False, affine_trainable = True)
        
        
    def forward(self, x):
        """
        Calculate the output of the model
        
        Parameters
        ----------
        x : th.tensor
            Sliding window in NN format
            
        Returns
        -------
        Output of the ETDKAN with x as input
        
        """
        
        batch_size = x.shape[0]
        
        x_complex = th.view_as_complex(x.reshape((batch_size, self.M + 1, 2)))
        x_abs     = th.abs(x_complex)
        
        x_out = self.KAN.forward(x_abs)
        x_out = th.view_as_complex(x_out.reshape((batch_size, self.M + 1, 2)))
        
        y = th.sum(x_out*x_complex, axis = 1)
        y = th.view_as_real(y)
        
        return y

    def set_symb(self, weight_simple = 0.5):
        """
        Approximate the B-splines by analytical functions with symbolic regression
        
        Parameters
        ----------
        weight_simple : float, optional
            Weight (from 0 to 1) that prioritizies simplicity (0) over performance (1) for the choice of the analytical function. The default is 0.5
        
        """
        
        lib = ['x', 'x^2', 'x^3', 'x^4', 'x^5', 'sinh', 'tanh', 'sin', 'abs']
        kn.add_symbolic('sinh', th.sinh, c=3)
        
        self.KAN.auto_symbolic(lib = lib, verbose = 0, weight_simple = weight_simple)
        self.symb = True
        
        self.y_symb_list = []
        
        for i in range(2*(self.M + 1)):
            self.y_symb_list.append(kn.ex_round(self.KAN.symbolic_formula()[0][i], 5))
        
        
    def get_symb(self):
        """
        If symb = True, return the symbolic form of the inner KAN
        
        """
        
        if self.symb:
            return self.y_symb_list
        else:
            print("KAN model is not symbolyc yet.")

    def get_param(self):
        """
        Return the model attributes (M, N, k, grid, symb)
        
        """
        return (self.M, self.N, self.k, self.grid, self.symb)

    def calcNFLOP(self):
        """
        Calculate the number of floating point operations (NFLOP) for a forward step of the model
        
        Returns
        -------
        NFLOPs : int (or list)
            NFLOP of the model (if symb = True, the output is a list [NFLOPs_min, NFLOPs_max])
        
        """
        if not(self.symb): 
            NFLOPs = 10*(self.M + 1) + 6*(self.M + 1) + 2*self.M + (9*self.k*(self.grid + 1.5*self.k) + 2*self.grid - 2.5*self.k + 3)*(3*self.N*(self.M + 1)) + (self.M + self.N + 1)*8 + self.N
        
        else:
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
                    i_sz = self.N
                    j_sz = self.M + 1
                else:
                    i_sz = 2*(self.M + 1)
                    j_sz = self.N
                
                for i in range(i_sz):
                    for j in range(j_sz):
                        func = self.KAN.symbolic_fun[l].funs_name[i][j]
                        NFLOPs_max += NFLOPs_estimation[func][1]
                        NFLOPs_min += NFLOPs_estimation[func][0]
            
            NFLOPs_min += 10*(self.M + 1) + 6*(self.M + 1) + 2*self.M
            NFLOPs_max += 10*(self.M + 1) + 6*(self.M + 1) + 2*self.M
            
            NFLOPs = [NFLOPs_min, NFLOPs_max]
        
        return NFLOPs
        
    
    def save(self, model_path):
        """
        Save the model coefficients
        
        Parameters
        ----------
        model_path : string
            Path where the file will be stored
        
        """
        
        self.KAN.saveckpt(model_path)
        
    def load(self, model_path):
        """
        Load a model coefficients
        
        Parameters
        ----------
        model_path : string
            Path where the loaded file is stored
        
        """
        
        self.KAN = self.KAN.loadckpt(model_path)