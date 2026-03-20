# -*- coding: utf-8 -*-
"""
======================================================================
Funções para aplicação, treinamento e verificação de desempenho da NN-DPD
======================================================================
"""

import numpy as np
import torch as th

from torch           import nn
from tqdm.notebook   import tqdm
from dpd.torchUtils  import createDatasets, slidingWindowMP
from dpd.models      import ARVTDNN, ETDNN, ETDKAN, MP

#%%
#To-do list

### Mais importantes
# - Criar exemplo de DPD e idenficador (concluindo)

### Menos importantes
# - Adicionar modelos de ARVTDKAN
# - Adicionar sumário de funções
# - Adicionar comentários e descrições de funções

#%%


def trainMP(sigIn, sigRef, paramTrain, paramModel):
    
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


def custom_prune(model, node_th=1e-2, edge_th=3e-2):
    model.attribute()
    model.prune_edge(edge_th, log_history=False)
    model.forward(model.cache_data)
    
    model.attribute()
    model.log_history('prune')
    
    model = model.prune_node(node_th, log_history=False)
    
    return model