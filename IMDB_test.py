#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  3 20:51:51 2021

@author: minming
"""


from src import model, utils
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import time, sys
from torchtext import data
from torchtext import datasets
import argparse
import random
import json
import torch.multiprocessing as mp

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def test(args, device):
    args.task_id = device[-1]
    if int(device[-1]) >= torch.cuda.device_count():
        device = device[:-1]+str(int(device[-1])%torch.cuda.device_count())
    
    target_addr = sys.path[0]+'/results'
    with open(target_addr+"/IMDB_bestmodel.json", 'rb') as handle:
        dat = handle.read()
        
    bestmodel = json.loads(dat)
    c, h = bestmodel['c'], bestmodel['h']
    res = {}
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
    
   
    UNK_IDX = TEXT.vocab.stoi[TEXT.unk_token]
    PAD_IDX = TEXT.vocab.stoi[TEXT.pad_token]
    
    criterion = nn.BCEWithLogitsLoss()
    
        
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
            torch.save(classifier.state_dict(), saved_model_dir+str(args.task_id)+'classifier_IMDB.pt')
  
  
        print(f'Epoch: {epoch+1} | Epoch Time: {epoch_min}m {epoch_sec:.2f}s')
        print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.2f}% | task {args.task_id}')
        print(f'\t Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc*100:.2f}% | task {args.task_id}')
        
    classifier.load_state_dict(torch.load(saved_model_dir+str(args.task_id)+'classifier_IMDB.pt'))
    test_loss, test_acc = utils.evaluateNLP(classifier, test_iterator, criterion)
    print(f'Test Loss: {test_loss:.3f} | Test Acc: {test_acc*100:.2f}% | task {args.task_id}')
    
    res['CNNSig'] = test_acc
    del classifier
    
    
    if args.RNN:
        # consider a bidirectional LSTM
        torch.cuda.empty_cache()
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

            start_time = time.time()
    
            train_loss, train_acc = utils.trainNLP(rnn, train_iterator, optimizer, criterion)
            valid_loss, valid_acc = utils.evaluateNLP(rnn, valid_iterator, criterion)
    
            end_time = time.time()

            epoch_mins, epoch_secs = utils.epoch_time(start_time, end_time)
    
            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                torch.save(rnn.state_dict(), saved_model_dir+str(args.task_id)+'IMDB_rnn.pt')
    
            print(f'Epoch: {epoch+1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')
            print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.2f}% | task {args.task_id}')
            print(f'\t Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc*100:.2f}% | task {args.task_id}')
            
        rnn.load_state_dict(torch.load(saved_model_dir+str(args.task_id)+'IMDB_rnn.pt'))
        print(f"Task {args.task_id} RNN model loaded successfully.")
        test_loss, test_acc = utils.evaluateNLP(rnn, test_iterator, criterion)

        print(f'Test Loss: {test_loss:.3f} | Test Acc: {test_acc*100:.2f}% | task {args.task_id}')
        res['RNN'] = test_acc
        del rnn
    with open(target_addr+'/IMDB_test'+str(args.task_id)+'.json', 'w') as fp:
        json.dump(res, fp)
    return



if __name__=="__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--depth', type=int, help='signature depth', required=True)
    parser.add_argument('--epochs', type=int, help='training epochs', required=True)
    parser.add_argument('--RNN', type=bool, help='implement RNN', default=False)
    parser.add_argument('--drop_out', type=float, help='drop out rate', default=0.2)
    parser.add_argument('--device_id', type=int, help='device id', default=0)
    args = parser.parse_args()
    
    mp.set_start_method('spawn')
    
    # due the GPU memory limitation for LSTM model, we can only parallelize 4 processes
    k = 4
    pool = mp.Pool(processes = k)
    with pool:
        pool.starmap(test, [(args, 'cuda:'+str(i)) for i in range(k)])
    
    test(args, 'cuda:4')
    
    accs_rnn = []
    accs_cnnsig = []
    target_addr = 'results'#sys.path[0]+'/results'
    for i in range(5):
        with open(target_addr+"/IMDB_test"+str(i)+'.json', 'rb') as f:
            dat = f.read()
        dat = json.loads(dat)
        accs_rnn.append(dat['RNN'])
        accs_cnnsig.append(dat['CNNSig'])
                
    print(f"RNN has mean {np.mean(accs_rnn)}, std {np.std(accs_rnn)}.")
    print(f"CNNSig has mean {np.mean(accs_cnnsig)}, std {np.std(accs_cnnsig)}.")
    
#python3 IMDB_test.py --depth 4 --epochs 15 --RNN True --drop_out 0.5