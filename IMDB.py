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
import time, sys
from torchtext import data
from torchtext import datasets
import random

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)




if __name__=="__main__":
    seed = 123
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

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_iterator, valid_iterator, test_iterator = data.BucketIterator.splits((train_data, 
                                                                                valid_data, 
                                                                                test_data),
                                                                                batch_size = BATCH_SIZE,
                                                                                sort_within_batch = True,
                                                                                device = device
                                                                                )
    m = 4
    classifier = model.CNNSigNLP(vocab_size=len(TEXT.vocab), out_dimension=2, 
                                 pad_idx=TEXT.vocab.stoi[TEXT.pad_token], sig_depth=m)
    

    print(f'The model has {count_parameters(classifier):,} trainable parameters')
    pretrained_embedding = TEXT.vocab.vectors
    print(f"pretrained embedding size: {pretrained_embedding.size()}")
    
    UNK_IDX = TEXT.vocab.stoi[TEXT.unk_token]
    PAD_IDX = TEXT.vocab.stoi[TEXT.pad_token]
    classifier.embed.weight.data[UNK_IDX] = torch.zeros(100)
    classifier.embed.weight.data[PAD_IDX] = torch.zeros(100)
    
    classifier = classifier.to(device)
    
    lr = 0.001
    optimizer = optim.Adam(classifier.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss().to(device)
    
    #optimizer.param_groups[0]['lr']=1e-4
    N_epochs = 5
    best_valid_loss = float('inf')
    best_valid_acc = -float('inf')
    
    target_addr = sys.path[0]+'/results'
    saved_model_dir = 'trained_model/'
    for epoch in range(N_epochs):
        star_time = time.time()

        train_loss, train_acc = utils.trainNLP(classifier, train_iterator, optimizer, criterion)
        valid_loss, valid_acc = utils.evaluateNLP(classifier, valid_iterator, criterion)
  
        end_time = time.time()

        epoch_min, epoch_sec = utils.epoch_time(star_time, end_time)
        
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(classifier.state_dict(), saved_model_dir+'classifier_IMDB.pt')
  
  
        print(f'Epoch: {epoch+1} | Epoch Time: {epoch_min}m {epoch_sec:.2f}s')
        print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.2f}%')
        print(f'\t Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc*100:.2f}%')
        
    test_loss, test_acc = utils.evaluateNLP(classifier, test_iterator, criterion)
    print(f'Test Loss: {test_loss:.3f} | Test Acc: {test_acc*100:.2f}%')
