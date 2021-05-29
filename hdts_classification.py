#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 12 13:37:54 2021

@author: minming
"""

import argparse
from tslearn.datasets import UCR_UEA_datasets
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import RidgeClassifierCV
from tqdm import tqdm
from src import model, utils
from src.utils import prepareDataLoader
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import sys
import time



_datasets = ['PEMS-SF', 
             'JapaneseVowels',
             'FingerMovements', 
             'FaceDetection', 
             'PhonemeSpectra', 
             'MotorImagery', 
             'Heartbeat'
             ] 
             


    
def padding(data):
    d = data[0].shape[0]
    maxlen = max([p.shape[1] for p in data])
    print("Maximun length for this dataset is {}.".format(maxlen))
    for i in range(len(data)):
        p = data[i].astype('float32')
        data[i] = np.append(p, [[0 for j in range(maxlen - p.shape[1])] for i in range(d)], axis=1)
        #print(p.shape)
  
    return data

if __name__ == '__main__':
    
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--depth', type=int, help='signature depth', required=True)
    parser.add_argument('--epochs', type=int, help='training epochs', required=True)
    parser.add_argument('--batch_size', type=int, help='batch size', required=True)
    parser.add_argument('--rocket', type=bool, help='implement ROCKET', default=False)
    
    args = parser.parse_args()
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    target_addr = sys.path[0]+'/results'
    f = open(target_addr+"/TrainningResult", "w")
    sys.stdout = f
    datasets = tqdm(_datasets, position=0, leave=True)
    
    for name in datasets:
        #load dataset
        print(f"=========Dataset: {name}==========")
        x_train, y_train, x_test, y_test = UCR_UEA_datasets().load_dataset(name)
        np.nan_to_num(x_train, copy=False)
        np.nan_to_num(x_test, copy=False)
        idx = np.random.permutation(x_train.shape[0]) # randomize data
        x_train, y_train = x_train[idx], y_train[idx]
        #x_train /= x_train.max() #somehow normalize the data scale
        
        y_train = LabelEncoder().fit_transform(y_train)
        y_test = LabelEncoder().fit_transform(y_test)
        
        datasets.set_description(f"dataset: {name} --- shape: {x_train.shape}")
        
        if args.rocket:
            t_rocket_start = time.time()
            from sktime.transformations.panel.rocket import Rocket
            #do rockect
            rocket = Rocket()
            rocket.fit(x_train)
            
            x_train_transform = rocket.transform(x_train)
            classifier = RidgeClassifierCV(alphas=np.logspace(-3, 3, 10), normalize=True)
            classifier.fit(x_train_transform, y_train)
            
            x_test_transform = rocket.transform(x_test)
            t_rocket_end = time.time()
            rocket_minute, rocket_second = utils.epoch_time(t_rocket_start, t_rocket_end)
            print(f"Rocket training time: {rocket_minute}m {round(rocket_second, 2)}s")
            print(f"Rocket: dataset--{name}, testing acc--{classifier.score(x_test_transform, y_test)}")
            
            
            
            
        
        batch, l, in_channels = x_train.shape
        batch_train = batch
        batch_valid = batch - batch_train
        out_dimension = max(y_train) + 1
        
        # hyper parameters      
        _cnnkernels = tqdm([(in_channels//gamma, h) for gamma in [1,2,3] for h in [1,2,3]], position=1, leave=False)
        layers = [256]
        sig_depth = args.depth
        epochs = args.epochs
        
        
        trainDataLoader = prepareDataLoader(x_train[:batch_train], 
                                            y_train[:batch_train], 
                                            l, 
                                            in_channels, 
                                            args.batch_size,
                                            device)
        
        testDataLoader = prepareDataLoader(x_test, 
                                           y_test, 
                                           l, 
                                           in_channels, 
                                           args.batch_size,
                                           device)
        validDataLoader = testDataLoader
        
        # grid search for optimal (c, h)
        best_valid_loss = float('inf')
        best_acc = 0
        saved_model_dir = 'trained_model/'
        lr = 0.001
        criterion = nn.CrossEntropyLoss().to(device)
        t_start = time.time()
        k = 5 # k-fold Cross Validation
        best_model_loss = float('inf')
        for (c, h) in _cnnkernels:
            _cnnkernels.set_description(f"CNN Kernel size: ({c}, {h})")
            # specify overlap by define stride
            #stride = None
            classifier = model.CNNSigFF(in_channels, out_dimension, 
                                        c, layers, sig_depth, 
                                        h=h, stride=(h, c), 
                                        out_channels=c).to(device)
            
            optimizer = optim.Adam(classifier.parameters(), lr=lr)
            
            cvLoss, cvAcc, cvEpochs = utils.CrossValidation(k,      # add for cross validation
                                                            x_train, 
                                                            y_train, 
                                                            classifier, 
                                                            optimizer, 
                                                            criterion, 
                                                            lr, 
                                                            epochs, 
                                                            args.batch_size, 
                                                            device)
            print(f"({c}, {h}): CV Loss: {round(cvLoss, 6)}---CV Acc: {round(cvAcc*100, 2)}%---CV Epochs: {cvEpochs}")
            if cvLoss < best_model_loss: # choose model based on CV
                best_c, best_h = c, h
                best_model_loss = cvLoss
                best_epochs = cvEpochs
            
            
        # train classifier with best model
        print(f"best model parameter: (c, h)={best_c}, {best_h}) --- best CV Loss: {round(best_model_loss, 6)}")
        classifier = model.CNNSigFF(in_channels, out_dimension, 
                                    best_c, layers, sig_depth, 
                                    h=best_h, stride=(best_h, best_c),
                                    out_channels=best_c).to(device)
        optimizer = optim.Adam(classifier.parameters(), lr=lr)
        for epoch in range(best_epochs):
            if epoch > 30: 
                optimizer.param_groups[0]['lr'] = .0001
            train_loss = utils.train(classifier, trainDataLoader, optimizer, criterion, device)
        torch.save(classifier.state_dict(), saved_model_dir+'classifer_'+name+'.net')
        
        t_end = time.time()
        epoch_minute, epoch_second = utils.epoch_time(t_start, t_end)
        print(f"training time: {epoch_minute}m {round(epoch_second, 2)}s")
        print(f"Train loss: {train_loss}")
        
        # test model
        acc_test = utils.acc(classifier, testDataLoader, device)
        print(f"CNNSig: dataset--{name}, testing acc--{round(acc_test*100, 2)}%")
    f.close()
            
