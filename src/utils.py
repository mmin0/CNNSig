#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 12 16:17:10 2021

@author: minming
"""
import torch
import numpy as np
#import torch.multiprocessing as mp
from .model import CNNSigNLP, RNN


def prepareDataLoader(x, y, l, in_channels, batch_size, device):
    x = torch.tensor(x, device=device, dtype=torch.float32).reshape(-1, 1, l, in_channels)
    y = torch.tensor(y, device=device, dtype=torch.long)
    DataLoader = torch.utils.data.TensorDataset(x, y)
    return torch.utils.data.DataLoader(DataLoader, 
                                       shuffle=True, 
                                       batch_size=batch_size)
    

def epoch_time(start_time, end_time):
    elap_time = end_time - start_time
    elap_min = elap_time//60
    elap_sec = elap_time % 60
    return elap_min, elap_sec


def missingValue(dat, p):
    """
    input:
        dat     - torch.tensor, data to be modified
        p       - float, missing probability
    """
    rand = np.random.uniform(size = dat.shape)
    dat[rand<p] = 0
    return dat


def train(model, dataloader, optimizer, criterion, device):
    """
    train model for one loop over dataloader
    input:
        model       - NN
        dataloarder - DataLoader
        criterion   - loss
        depth       - signature depth
        device      - device, cuda or cpu
    """
    epoch_loss = 0
    model.train() # set model to train mode
    i = 0
    for i, data in enumerate(dataloader):
        x_train, y_train = data
        optimizer.zero_grad()
        y_pred = model(x_train)
        loss = criterion(y_pred, y_train)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        i+=1
    return epoch_loss/len(dataloader)


def evaluate(model, dataloader, criterion, device):
    """
    test model for one loop over dataloader
    input:
        model       - NN
        dataloarder - DataLoader
        criterion   - loss
        depth       - signature depth
        device      - device, cuda or cpu
    """
    epoch_loss = 0
    model.eval() # set model to train mode
    i = 0
    for i, data in enumerate(dataloader):
        x_train, y_train = data
        y_pred = model(x_train)
        loss = criterion(y_pred, y_train)
        loss.backward()
        epoch_loss += loss.item()
        i+=1
    return epoch_loss/len(dataloader)

def acc(model, dataloader, device):
    count = 0
    l = 0
    for i, data in enumerate(dataloader):
        x, y = data
        y_pred = model(x)
        count += (torch.max(y_pred, 1)[1]==y).sum().item()
        l += len(y)
    #return sum(y_pred==y_true)
    return count/l


def CrossValidation(k, x, y, model, optimizer, criterion, lr, epochs, batch_size, device):
    """
    input:
        k       - k-fold CV
        params  - (c, h)
        x, y    - data
    """
    batch, l, in_channels = x.shape
    cv_batch = batch // k
    cv_loss = 0
    cv_acc = 0
    cv_epochs = 0
    for i in range(k):
        # need to initialize model parameters
        model.initialize_weights()
        test_x, test_y = x[i*cv_batch:(i+1)*cv_batch], y[i*cv_batch:(i+1)*cv_batch]
        train_x, train_y =  np.append(x[:i*cv_batch], x[(i+1)*cv_batch:]), np.append(y[:i*cv_batch], y[(i+1)*cv_batch:])
        testDataLoader = prepareDataLoader(test_x, test_y, l, in_channels, batch_size, device)
        trainDataLoader = prepareDataLoader(train_x, train_y, l, in_channels, batch_size, device)
        best_valid_loss = float('inf')
        best_epoch = 0
        optimizer.param_groups[0]['lr'] = lr
        for epoch in range(epochs):
            if epoch > 30:
                optimizer.param_groups[0]['lr'] /= 10
            train(model, trainDataLoader, optimizer, criterion, device)
            valid_loss = evaluate(model, testDataLoader, criterion, device)
            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                best_epoch = epoch
                best_acc = acc(model, testDataLoader, device)
        cv_loss += best_valid_loss #evaluate(model, testDataLoader, criterion, device)
        cv_acc += best_acc
        
        cv_epochs += best_epoch # control overfitting problem
    return cv_loss/k, cv_acc/k, cv_epochs//k

'''
def CrossValidationHelper(model, trainDataLoader, testDataLoader, optimizer, criterion, epochs, device):
    best_valid_loss = float('inf')
    best_epoch = 0
    for epoch in range(epochs):
        if epoch > 30:
            optimizer.param_groups[0]['lr'] /= 10
        train(model, trainDataLoader, optimizer, criterion, device)
        valid_loss = evaluate(model, testDataLoader, criterion, device)
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            best_epoch = epoch
            best_valid_acc = acc(model, testDataLoader, device)
    return best_valid_loss, best_valid_acc, best_epoch
    


def CrossValidationParallel(k, x, y, model, optimizer, criterion, epochs, batch_size, device):
    """
    input:
        k       - k-fold CV
        params  - (c, h)
        x, y    - data
    """
    assert k > torch.cuda.device_count(), "# of folds should be smaller than # of cuda nodes"
    batch, l, in_channels = x.shape
    cv_batch = batch // k
    
    # start parallel computing
    #model.share_memory()
    mp.set_start_method('spawn')
    processes = []
    for i in range(k):
        test_x, test_y = x[i*cv_batch:(i+1)*cv_batch], y[i*cv_batch:(i+1)*cv_batch]
        train_x, train_y =  np.append(x[:i*cv_batch], x[(i+1)*cv_batch:]), np.append(y[:i*cv_batch], y[(i+1)*cv_batch:])
        testDataLoader = prepareDataLoader(test_x, test_y, l, in_channels, batch_size, device)
        trainDataLoader = prepareDataLoader(train_x, train_y, l, in_channels, batch_size, device)
        
        
        p = mp.Process(target = CrossValidationHelper, 
                       args=(model, trainDataLoader, testDataLoader,
                             optimizer, criterion, epochs, device))
        
        p.start()
        processes.append(p)
    for p in processes:
        p.join()
    return
    '''

def binary_acc(preds, y):
    rounded_preds = torch.round(torch.sigmoid(preds))
    correct = (rounded_preds==y).float()
    return correct.sum()/len(correct)

def trainNLP(model, iterator, optimizer, criterion):
    epoch_loss = 0
    epoch_acc = 0
    model.train()
    for batch in iterator:
        optimizer.zero_grad()
        text, text_lengths = batch.text
        if isinstance(model, CNNSigNLP):
            preds = model(text).squeeze(1)
        elif isinstance(model, RNN):
            preds = model(text, text_lengths).squeeze(1)
        
        loss = criterion(preds, batch.label)
        acc = binary_acc(preds, batch.label)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        epoch_acc += acc.item()
    
    return epoch_loss/len(iterator), epoch_acc/len(iterator)
 

def evaluateNLP(model, iterator, criterion):
    epoch_loss = 0
    epoch_acc = 0

    model.eval()
    with torch.no_grad():
        for batch in iterator:
            text, text_lengths = batch.text
            if isinstance(model, CNNSigNLP):
                preds = model(text).squeeze(1)
            elif isinstance(model, RNN):
                preds = model(text, text_lengths).squeeze(1)
            loss = criterion(preds, batch.label)
            acc = binary_acc(preds, batch.label)

            epoch_loss += loss.item()
            epoch_acc += acc.item()
    return epoch_loss/len(iterator), epoch_acc/len(iterator)