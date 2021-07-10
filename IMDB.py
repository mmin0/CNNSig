#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 26 15:11:12 2021

@author: minming
"""

from src import model, utils
import torch
import torch.nn as nn
import torch.optim as optim
import time, sys, os
from torchtext import data
from torchtext import datasets
import argparse
import random
from tqdm import tqdm
import json

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)




if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--depth', type=int, help='signature depth', required=True)
    parser.add_argument('--epochs', type=int, help='training epochs', required=True)
    parser.add_argument('--RNN', type=bool, help='implement RNN', default=False)
    parser.add_argument('--drop_out', type=float, help='drop out rate', default=0.2)
    parser.add_argument('--device_id', type=int, help='device id', default=0)
    args = parser.parse_args()
    
    seed = random.randint(0,10000)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

    TEXT = data.Field(tokenize='spacy', tokenizer_language = 'en_core_web_sm', include_lengths=True)
    LABEL = data.LabelField(dtype = torch.float32)

    train_data, test_data = datasets.IMDB.splits(TEXT, LABEL)
    
    train_data, valid_data = train_data.split(random_state=random.seed(seed))
    print(f'Training size = {len(train_data)}')
    print(f'Validation size = {len(valid_data)}')
    print(f'Testing size = {len(test_data)}')
    
    MAX_VOCAB_SIZE = 25000

    TEXT.build_vocab(train_data, max_size = MAX_VOCAB_SIZE, vectors = 'glove.6B.100d', unk_init=torch.Tensor.normal_)

    LABEL.build_vocab(train_data)
    
    BATCH_SIZE = 64

    device = torch.device('cuda:'+str(args.device_id) if torch.cuda.is_available() else 'cpu')

    train_iterator, valid_iterator, test_iterator = data.BucketIterator.splits((train_data, 
                                                                                valid_data, 
                                                                                test_data),
                                                                                batch_size = BATCH_SIZE,
                                                                                sort_within_batch = True,
                                                                                device = device
                                                                                )
    m = args.depth
    layers = [256]
    N_epochs = args.epochs
        
    
    in_channels = 100
    _cnnkernels = tqdm([(in_channels//gamma, h) for gamma in [1,2,3] for h in [1,2,3]], position=1, leave=False)
    
    UNK_IDX = TEXT.vocab.stoi[TEXT.unk_token]
    PAD_IDX = TEXT.vocab.stoi[TEXT.pad_token]
    
    best_model_valid_loss = float('inf')
    best_model_valid_acc = -float('inf')
    criterion = nn.BCEWithLogitsLoss()
    
    for (c, h) in _cnnkernels:
        _cnnkernels.set_description(f"CNN Kernel size: ({c}, {h})")
        if h>1:
            classifier = model.CNNSigNLP(vocab_size=len(TEXT.vocab), out_dimension=1, 
                                         pad_idx=TEXT.vocab.stoi[TEXT.pad_token], c=c,
                                         layers=layers, h=h, stride=(h-1, c),
                                         out_channels=c//10, sig_depth=m, dropout=args.drop_out)
        else:
            classifier = model.CNNSigNLP(vocab_size=len(TEXT.vocab), out_dimension=1, 
                                         pad_idx=TEXT.vocab.stoi[TEXT.pad_token], c=c,
                                         layers=layers, h=h, stride=(h, c),
                                         out_channels=c//10, sig_depth=m, dropout=args.drop_out)
    
        print(f"=========CNN Kernel size: ({c}, {h})===========")
        print(f'The model has {count_parameters(classifier):,} trainable parameters')
        pretrained_embedding = TEXT.vocab.vectors
        print(f"pretrained embedding size: {pretrained_embedding.size()}")
    
    
        classifier.embed.weight.data[UNK_IDX] = torch.zeros(100)
        classifier.embed.weight.data[PAD_IDX] = torch.zeros(100)
    
        classifier = classifier.to(device)
    
        lr = 0.001
        optimizer = optim.Adam(classifier.parameters(), lr=lr)
        
    
        #optimizer.param_groups[0]['lr']=1e-4
        
        best_valid_loss = float('inf')
        best_valid_acc = -float('inf')
    
        target_addr = sys.path[0]+'/results'
        saved_model_dir = 'trained_model/'
        for epoch in range(N_epochs):
            if epoch>=10:
                optimizer.param_groups[0]['lr'] /= 10
            star_time = time.time()

            train_loss, train_acc = utils.trainNLP(classifier, train_iterator, optimizer, criterion)
            valid_loss, valid_acc = utils.evaluateNLP(classifier, valid_iterator, criterion)
  
            end_time = time.time()

            epoch_min, epoch_sec = utils.epoch_time(star_time, end_time)
        
            if valid_loss < best_valid_loss:
                best_h, best_c = (h, c)
                best_valid_loss = valid_loss
                torch.save(classifier.state_dict(), saved_model_dir+'classifier_IMDB_.pt')
  
  
            print(f'Epoch: {epoch+1} | Epoch Time: {epoch_min}m {epoch_sec:.2f}s')
            print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.2f}%')
            print(f'\t Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc*100:.2f}%')
        
        classifier.load_state_dict(torch.load(saved_model_dir+'classifier_IMDB_.pt'))
        test_loss, test_acc = utils.evaluateNLP(classifier, test_iterator, criterion)
        print(f'Test Loss: {test_loss:.3f} | Test Acc: {test_acc*100:.2f}%')
        
        if best_valid_loss < best_model_valid_loss:
            best_model_h, best_model_c = (best_h, best_c)
            best_model_valid_loss = best_valid_loss
            #torch.save(classifier.state_dict(), saved_model_dir+'classifier_IMDB.pt')
            os.rename(saved_model_dir+'classifier_IMDB_.pt', saved_model_dir+'classifier_IMDB.pt')
    print(f"Best mdoel: (h, c)= ({best_model_h}, {best_model_c})")
    bestmodel = {'h': best_model_h, 'c':best_model_c}
    with open(target_addr+'/IMDB_bestmodel.json', 'w') as fp:
        json.dump(bestmodel, fp)
    
    
    if args.RNN:
        # consider a bidirectional LSTM
        print("=====Using RNN======")
        INPUT_DIM = len(TEXT.vocab)
        EMBEDDING_DIM = 100
        HIDDEN_DIM = 256
        OUTPUT_DIM = 1
        N_LAYERS = 2
        BIDIRECTIONAL = True
        DROPOUT = args.drop_out
        PAD_IDX = TEXT.vocab.stoi[TEXT.pad_token]

        rnn = model.RNN(INPUT_DIM, 
                        EMBEDDING_DIM, 
                        HIDDEN_DIM, 
                        OUTPUT_DIM, 
                        N_LAYERS, 
                        BIDIRECTIONAL, 
                        DROPOUT, 
                        PAD_IDX)
        
        rnn = rnn.to(device)
        optimizer = optim.Adam(rnn.parameters(), lr=lr)
        
        for epoch in range(N_epochs):
            if epoch>=10:
                optimizer.param_groups[0]['lr'] /= 10
            start_time = time.time()
    
            train_loss, train_acc = utils.trainNLP(rnn, train_iterator, optimizer, criterion)
            valid_loss, valid_acc = utils.evaluateNLP(rnn, valid_iterator, criterion)
    
            end_time = time.time()

            epoch_mins, epoch_secs = utils.epoch_time(start_time, end_time)
    
            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                torch.save(rnn.state_dict(), saved_model_dir+'IMDB_rnn.pt')
    
            print(f'Epoch: {epoch+1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')
            print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.2f}%')
            print(f'\t Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc*100:.2f}%')
            
        rnn.load_state_dict(torch.load(saved_model_dir+'IMDB_rnn.pt'))

        test_loss, test_acc = utils.evaluateNLP(rnn, test_iterator, criterion)

        print(f'Test Loss: {test_loss:.3f} | Test Acc: {test_acc*100:.2f}%')
        
        
        