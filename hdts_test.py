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
import json
import torch.multiprocessing as mp



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

def test(args, device):
    
    args.task_id = device[-1]
    if int(device[-1]) >= torch.cuda.device_count():
        device = device[:-1]+str(int(device[-1])%torch.cuda.device_count())
    
    target_addr = sys.path[0]+'/results'
    with open(target_addr+"/hdts_bestmodels.txt", 'rb') as handle:
        data = handle.read()
    best_models = json.loads(data)
    
    f = open(target_addr+"/TrainningResult", "w")
    sys.stdout = f
    datasets = tqdm(_datasets, position=0, leave=True)
    
    res = {}
    
    for name in datasets:
        #load dataset
        res[name] = {}
        print(f"=========Dataset: {name}==========")
        x_train, y_train, x_test, y_test = UCR_UEA_datasets().load_dataset(name)
        np.nan_to_num(x_train, copy=False)
        np.nan_to_num(x_test, copy=False)
        idx = np.random.permutation(x_train.shape[0]) # randomize data
        x_train, y_train = x_train[idx], y_train[idx]
        #x_train /= x_train.max() #somehow normalize the data scale
        
        if args.miss_value > 0:
            x_train = utils.missingValue(x_train, args.miss_value)
            x_test = utils.missingValue(x_test, args.miss_value)
        
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
            
            res[name]['rocket'] = classifier.score(x_test_transform, y_test)
            
            
            
        
        batch, l, in_channels = x_train.shape
        batch_train = batch
        out_dimension = max(y_train) + 1
        
        # hyper parameters      
        
        layers = [256]
        sig_depth = args.depth
        
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
        
        # grid search for optimal (c, h)
        lr = 0.001
        criterion = nn.CrossEntropyLoss().to(device)
        t_start = time.time()
        
        ###############
        c = best_models[name]["c"]
        h = best_models[name]["h"]
        epochs = best_models[name]["epochs"]
        ##############
        # specify overlap by define stride
        #stride = None
            
        # classifier with best model
        classifier = model.CNNSigFF(in_channels, out_dimension, 
                                    c, layers, sig_depth, 
                                    h=h, stride=(h, c),
                                    out_channels=c).to(device)
        optimizer = optim.Adam(classifier.parameters(), lr=lr)
        for epoch in range(epochs):
            if epoch > 30:
                optimizer.param_groups[0]['lr'] /= 10
            train_loss = utils.train(classifier, trainDataLoader, optimizer, criterion, device)
        #torch.save(classifier.state_dict(), saved_model_dir+'classifer_'+name+'.net')
        
        t_end = time.time()
        epoch_minute, epoch_second = utils.epoch_time(t_start, t_end)
        print(f"training time: {epoch_minute}m {round(epoch_second, 2)}s")
        print(f"Train loss: {train_loss}")
        
        # test model
        acc_test = utils.acc(classifier, testDataLoader, device)
        print(f"CNNSig: dataset--{name}, testing acc--{round(acc_test*100, 2)}%")
        res[name]['CNNSig'] =  acc_test
    f.close()
    
    with open(target_addr+'/test'+str(args.task_id)+'.json', 'w') as fp:
        json.dump(res, fp)
    

if __name__ == '__main__':
    
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument('--depth', type=int, help='signature depth', required=True)
    #parser.add_argument('--epochs', type=int, help='training epochs', required=True)
    parser.add_argument('--batch_size', type=int, help='batch size', required=True)
    parser.add_argument('--rocket', type=bool, help='implement ROCKET', default=False)
    
    parser.add_argument('--miss_value', type=float, help='ratio of missing value', default=0)
    parser.add_argument('--task_id', type=int, help='id of task', required=False)
    
    args = parser.parse_args()
    #device = torch.device('cuda:'+str(args.gpu_id) if torch.cuda.is_available() else 'cpu')
    
    mp.set_start_method('spawn')
    pool = mp.Pool(processes = 5)
    with pool:
        pool.starmap(test, [(args, 'cuda:'+str(i)) for i in range(5)])
    '''
    accs_rocket = {}
    accs_cnnsig = {}
    target_addr = 'results'#sys.path[0]+'/results'
    for i in range(5):
        with open(target_addr+"/test"+str(i)+'.json', 'rb') as f:
            dat = f.read()
        dat = json.loads(dat)
        for name in _datasets:
            try:
                accs_rocket[name].append(dat[name]['rocket'])
            except KeyError:
                accs_rocket[name] = [dat[name]['rocket']]
            
            try:
                accs_cnnsig[name].append(dat[name]['CNNSig'])
            except KeyError:
                accs_cnnsig[name] = [dat[name]['CNNSig']]
                
    for name in _datasets:
        print(f"ROCKET {name} datasets has mean {np.mean(accs_rocket[name])}, std {np.std(accs_rocket[name])}.")
        print(f"CNNSig {name} datasets has mean {np.mean(accs_cnnsig[name])}, std {np.std(accs_cnnsig[name])}.")
    
                