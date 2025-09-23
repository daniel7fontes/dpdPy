# -*- coding: utf-8 -*-
"""
======================================================================
Funções para aplicação, treinamento e verificação de desempenho da NN-DPD
======================================================================
"""

import numpy as np
import torch as th
import kan   as kn

from tqdm.notebook   import tqdm
from torch           import nn

from optic_private.torchDSP   import pnorm
from optic_private.torchUtils import memoryLessDataSet, MLP, ETDNN, EKAN, slidingWindowDataSet, fitFilterNN

def NN_training(sigIn, sigRef, param):
    
    layers        = param.layers
    divByL        = param.divByL
    trainTestFrac = param.trainTestFrac
    batch_size    = param.batch_size
    shuffle       = param.shuffle
    includeMemory = param.includeMemory
    Ntaps         = param.Ntaps
    K             = param.K
    augment       = param.augment
    
    lr            = param.lr
    epochs        = param.epochs
    activation    = param.activation
    
    pgrsBar       = param.pgrsBar
    directLearn   = param.directLearn
    device        = param.device
    
    envelope      = param.envelope
    
    activations = {'leaky_relu': nn.LeakyReLU(), 'relu': nn.ReLU(), 'sigmoid': nn.Sigmoid(), 'tanh': nn.Tanh()}   
        
    # Define neural network (MLP) model
    if not(envelope):
        DPD_model = MLP(layers, activation = activations[activation] ).to(device)
    else:
        DPD_model = ETDNN(layers, activation = activations[activation] ).to(device)
    
    
    loss_fn = nn.MSELoss()
    optimizer = th.optim.Adam(DPD_model.parameters(), lr = lr)
       
    trainLoss = np.zeros(epochs)
    testLoss  = np.zeros(epochs)
    
    if directLearn:
        train_dataloader, test_dataloader = createDatasets(sigRef, sigRef, divByL, trainTestFrac,\
                                                           batch_size, includeMemory, Ntaps, K, device, shuffle = shuffle, augment=augment)
        numBatches_train = len(train_dataloader)
        numBatches_test  = len(test_dataloader)
        
        RoFChannel_model = param.RoFChannel_model
        
        for p in RoFChannel_model.parameters():
            p.requires_grad = False
        
        for t in tqdm(range(epochs), disable = not(pgrsBar)):   
            # Training
            DPD_model.train()
            trainLoss[t] = 0
            
            for batch, (_, batch_data) in enumerate(train_dataloader.items()):
                X = batch_data['input']
                y = batch_data['label']
                
                # Compute prediction error        
                chInput = DPD_model(X)                
                chInput = th.view_as_complex(chInput)            
                chOutput = th.view_as_real(fitFilterNN(chInput, RoFChannel_model, Ntaps, K, 1, len(chInput), predict=False))

                loss = loss_fn(chOutput, y)
                trainLoss[t] += loss.item()
                
                # Backpropagation
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            trainLoss[t] /= numBatches_train            
            
            # Validation      
            DPD_model.eval()
            testLoss[t] = 0
                    
            with th.no_grad():
                for batch, (_, batch_data) in enumerate(test_dataloader.items()):
                    X = batch_data['input']
                    y = batch_data['label']  
     
                    # Compute prediction error        
                    chInput = DPD_model(X)           
                    chInput = th.view_as_complex(chInput)
                    chOutput = th.view_as_real(fitFilterNN(chInput, RoFChannel_model, Ntaps, K, 1, len(chInput)))
                    
                    loss = loss_fn(chOutput, y)
                    testLoss[t] += loss.item()
        
            testLoss[t] /= numBatches_test   
       
            if (t+1) % 100 == 0:
                for g in optimizer.param_groups:
                    g['lr'] = g['lr']/1.25

    
    else:
        train_dataloader, test_dataloader = createDatasets(sigRef, sigIn, divByL, trainTestFrac,\
                                                           batch_size, includeMemory, Ntaps, K, device, shuffle = shuffle, augment=augment)
        numBatches_train = len(train_dataloader)
        numBatches_test  = len(test_dataloader)
        
        for t in tqdm(range(epochs), disable = not(pgrsBar)): 
            # Training
            DPD_model.train()
            trainLoss[t] = 0
            
            for batch, (_, batch_data) in enumerate(train_dataloader.items()):
                X = batch_data['input']
                y = batch_data['label']
            
                # Compute prediction error
                pred = DPD_model(X)
                loss = loss_fn(pred, y)
                
                # Backpropagation
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                trainLoss[t] += loss.item()
            
            trainLoss[t] /= numBatches_train
            
            # Validation
            DPD_model.eval()
            testLoss[t] = 0
            
            with th.no_grad():
                for batch, (_, batch_data) in enumerate(test_dataloader.items()):
                    X = batch_data['input']
                    y = batch_data['label']
                    
                    pred = DPD_model(X)
                    loss = loss_fn(pred, y)
                    
                    testLoss[t] += loss.item()
                
            testLoss[t] /= numBatches_test
            
            if (t+1)%100 == 0:
                for g in optimizer.param_groups:
                    g['lr'] = g['lr']/2

        
    return DPD_model, trainLoss, testLoss


def batch_data(data_input, data_label, batch_size, shuffle = False):
    N = data_input.shape[0]
    num_batches = int(np.floor(N / batch_size))

    data_dic = {}
    
    for b in range(num_batches):
        index = np.arange(b*batch_size, (b+1)*batch_size, dtype = int)
        
        if shuffle:
            np.random.shuffle(index)
        
        data_dic[f"batch_{b}"] = {"input" : data_input[index,:], 
                                  "label" : data_label[index,:]}
    if num_batches*batch_size < N:
        index = np.arange((b+1)*batch_size, N, dtype = int)
        
        if shuffle:
            np.random.shuffle(index)
        
        data_dic[f"batch_{b+1}"] = {"input" : data_input[index,:], 
                                    "label" : data_label[index,:]}
        
    return data_dic


def createDatasets(
    sigIn,
    sigRef,
    divByL,
    trainTestFrac,
    batch_size,
    includeMemory,
    Ntaps,
    K,
    device,
    shuffle = False,
    augment = False
):
    sig_in  = pnorm(sigIn[0 : len(sigIn) // divByL])
    sig_ref = pnorm(sigRef[0 : len(sigRef) // divByL])

    # Create the dataset
    indx_train = th.arange(0, int(trainTestFrac * len(sig_in)))
    indx_test = th.arange(int(trainTestFrac * len(sig_in)), len(sig_in))

    sig_train = sig_in[indx_train]  # get signal amplitude samples (L,)
    sig_test  = sig_in[indx_test]  # get signal amplitude samples (L,)

    indy_train = th.arange(0, int(trainTestFrac * len(sig_ref)))
    indy_test = th.arange(int(trainTestFrac * len(sig_ref)), len(sig_ref))
       
    if includeMemory:
        train_dataset = slidingWindowDataSet(
            sig_ref[indy_train], sig_train, Ntaps, K, augment=augment
        )
        test_dataset = slidingWindowDataSet(
            sig_ref[indy_test], sig_test, Ntaps, K, augment=augment
        )

    else:
        train_dataset = memoryLessDataSet(sig_ref[indy_train], sig_train, K, augment = augment)
        test_dataset  = memoryLessDataSet(sig_ref[indy_test], sig_test, K, augment = augment)
        
    
    # Train dataloader
    train_inputs = th.empty((0, (2+K)*Ntaps), device = device) if augment else th.empty((0, 2*Ntaps), device = device)
    train_labels = th.empty((0, 2), device = device)
        
    for data, label in train_dataset:
        train_inputs = th.cat((train_inputs, data.reshape(1, -1).to(device)), dim = 0)
        train_labels = th.cat((train_labels, label.reshape(1, -1).to(device)), dim = 0)
    
    batch_train = batch_size if batch_size <= train_inputs.shape[0] else train_inputs.shape[0]
    train_dataloader = batch_data(train_inputs, train_labels, batch_train, shuffle)
    
    # Test dataloader
    test_inputs = th.empty((0, (2+K)*Ntaps), device = device) if augment else th.empty((0, 2*Ntaps), device = device)
    test_labels = th.empty((0, 2), device = device)
     
    for data, label in test_dataset:
        test_inputs = th.cat((test_inputs, data.reshape(1, -1).to(device)), dim = 0)
        test_labels = th.cat((test_labels, label.reshape(1, -1).to(device)), dim = 0)

    
    batch_test = batch_size if batch_size <= test_inputs.shape[0] else test_inputs.shape[0]
    test_dataloader = batch_data(test_inputs, test_labels, batch_test, shuffle)
    
    return train_dataloader, test_dataloader


def custom_prune(model, node_th=1e-2, edge_th=3e-2):
    model.attribute()
    model.prune_edge(edge_th, log_history=False)
    #model.forward(model.cache_data)
    
    model.attribute()
    #model.log_history('prune')
    
    model = model.prune_node(node_th, log_history=False)
    
    return model


def KAN_training(sigIn, sigRef, param, RoFChannel_model = None):
    
    layers        = param.layers
    k             = param.k
    grid          = param.grid
    
    divByL        = param.divByL
    trainTestFrac = param.trainTestFrac
    batch_size    = param.batch_size
    shuffle       = param.shuffle
    includeMemory = param.includeMemory
    Ntaps         = param.Ntaps
    K             = param.K
    augment       = param.augment
    
    lr            = param.lr
    epochs        = param.epochs
    
    seed          = param.seed
    pgrsBar       = param.pgrsBar
    directLearn   = param.directLearn
    device        = param.device
    
    pruning_epochs = param.pruning_epochs
    
    envelope      = param.envelope
    
    # Define neural network (KAN) model
    if not(envelope):
        DPD_model = kn.KAN(width = layers, grid = grid, k = k, seed = seed, device = device, auto_save=False)
    else:
        DPD_model = EKAN(layers, grid, k, seed, device)
    
    loss_fn = nn.MSELoss()
    optimizer = th.optim.Adam(DPD_model.parameters(), lr = lr)
    
    trainLoss = np.zeros(epochs)
    testLoss  = np.zeros(epochs)
    
    if directLearn:
        train_dataloader, test_dataloader = createDatasets(sigRef, sigRef, divByL, trainTestFrac,\
                                                           batch_size, includeMemory, Ntaps, K, device, shuffle = shuffle, augment=augment)
        numBatches_train = len(train_dataloader)
        numBatches_test  = len(test_dataloader)
        
        RoFChannel_model = param.RoFChannel_model
        
        for p in RoFChannel_model.parameters():
            p.requires_grad = False
        
        for t in tqdm(range(epochs), disable = not(pgrsBar)):   
           
            # Pruning, if indicated
            if t in pruning_epochs:
                if envelope:
                    DPD_model.KAN = custom_prune(DPD_model.KAN)
                else:
                    DPD_model = custom_prune(DPD_model)
                    
            # Training
            DPD_model.train()
            trainLoss[t] = 0
            
            for batch, (_, batch_data) in enumerate(train_dataloader.items()):
                X = batch_data['input']
                y = batch_data['label']
                
                # Compute prediction error        
                chInput  = DPD_model(X)                
                chInput  = th.view_as_complex(chInput)            
                chOutput = th.view_as_real(fitFilterNN(chInput, RoFChannel_model, Ntaps, K, 1, len(chInput), predict=False))

                loss = loss_fn(chOutput, y)
                trainLoss[t] += loss.item()
                
                # Backpropagation
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            
            trainLoss[t] /= numBatches_train
            
            # Validation      
            DPD_model.eval()
            testLoss[t] = 0
            
            with th.no_grad():
                for batch, (_, batch_data) in enumerate(test_dataloader.items()):
                    X = batch_data['input']
                    y = batch_data['label']  
                    # Compute prediction error        
                    chInput = DPD_model(X)           
                    
                    chInput  = th.view_as_complex(chInput)
                    chOutput = th.view_as_real(fitFilterNN(chInput, RoFChannel_model, Ntaps, K, 1, len(chInput)))
                    
                    loss = loss_fn(chOutput, y)
                    testLoss[t] += loss.item()
                    
            testLoss[t] /= numBatches_test
            
            if (t+1)%100 == 0:
                for g in optimizer.param_groups:
                    g['lr'] = g['lr']/1.25
            
            if t % 50 == 0:
                DPD_model.update_grid(X)
    
        
    else:
        train_dataloader, test_dataloader = createDatasets(sigRef, sigIn, divByL, trainTestFrac,\
                                                           batch_size, includeMemory, Ntaps, K, device, shuffle = shuffle, augment=augment)
        numBatches_train = len(train_dataloader)
        numBatches_test  = len(test_dataloader)
        
        for t in tqdm(range(epochs), disable = not(pgrsBar)):             
            
            # Training
            DPD_model.train()
            trainLoss[t] = 0
            
            for batch, (_, batch_data) in enumerate(train_dataloader.items()):
                X = batch_data['input']
                y = batch_data['label']
            
                # Compute prediction error
                pred = DPD_model(X)
                loss = loss_fn(pred, y)
                
                # Backpropagation
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                trainLoss[t] += loss.item()
            
            trainLoss[t] /= numBatches_train
            
            if trainLoss[t] > 10:
                break
            
            # Validation
            DPD_model.eval()
            testLoss[t] = 0
            
            with th.no_grad():
                for batch, (_, batch_data) in enumerate(test_dataloader.items()):
                    X = batch_data['input']
                    y = batch_data['label']
                    
                    pred = DPD_model(X)
                    loss = loss_fn(pred, y)
                    
                    testLoss[t] += loss.item()
                
            testLoss[t] /= numBatches_test
            
            # Pruning, if indicated
            if t in pruning_epochs:
                if envelope:
                    DPD_model.KAN = custom_prune(DPD_model.KAN)
                else:
                    DPD_model = custom_prune(DPD_model)
            
            if t % 50 == 0:
                DPD_model.update_grid(X)
    
            if (t+1)%100 == 0:
                for g in optimizer.param_groups:
                    g['lr'] = g['lr']/2
        
    return DPD_model, trainLoss, testLoss