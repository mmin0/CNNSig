
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  3 14:21:47 2021

@author: minming
"""

import torch
import signatory
import torch.nn as nn
import torch.nn.functional as F


class View(nn.Module):
    def __init__(self, shape):
      super(View, self).__init__()
      self.shape = shape

    def forward(self, x):
      return x.view(*self.shape)
  

# we define a CNNSig feedforward classifier by this CNNSigFF
class CNNSigFF(nn.Module):
    def __init__(self, in_channels, out_dimension, c, layers, sig_depth, dropout=0.2, h=1, out_channels=None, stride=None, include_time=True):
        """
        input:
            in_channels     - int, dimension of input sequence
            out_dimension   - int, number of classes
            c               - int, width of the Convolutional Kernel
            layers          - tuple of int, feedforward hidden layers
            sig_depth       - int, depth of signature
            dropout         - float, dropout rate
            h               - int, height of Convolutional Kernel (in the time)
            out_channels    - int, out channels of CNN2D layer
            stride          - (int, int), stride size
            include_time    - bool, whether to include time dimension
        """
        super(CNNSigFF, self).__init__()
        if not stride:
            stride = (h,c) # non-overlap convolution
        if not out_channels:
            out_channels = c
        self.conv2d1 = torch.nn.Conv2d(1, out_channels, (h, c), stride=stride)    
        #self.BatchNorm = torch.nn.BatchNorm2d(out_channels, affine=True)
        self.channels = (in_channels-c)//stride[1]+1 # dimension of the low dimensional paths
        
        self.augment = signatory.Augment(in_channels=in_channels, 
                                         layer_sizes=(), 
                                         kernel_size=1,
                                         include_time=include_time) # include additional time dimension
        sig_channels = signatory.signature_channels(self.channels+include_time,
                                                    depth=sig_depth)
        self.signature = signatory.Signature(depth=sig_depth)
        self.v2 = View([-1, out_channels*sig_channels])
        self.linear = nn.ModuleList([nn.Linear(out_channels*sig_channels, layers[0])])
        for i in range(len(layers)-1):
            self.linear.append(nn.Linear(layers[i], layers[i+1]))
        self.linear.append(nn.Linear(layers[len(layers)-1], out_dimension))
        self.dropout = torch.nn.Dropout(dropout)
        self.initialize_weights()

    def forward(self, x):
        """
        input:
            x - tensor, sequential data
        """
        #x = self.BatchNorm(x)
        x = self.conv2d1(x)
        
        T = x.size()[2]
        x = x.view(-1, T, self.channels)
        x = self.augment(x) # augment with time dimension
        x = self.signature(x, basepoint=True)
        x = self.v2(x)
        # question: does the v2 will trace back to the correct structure, i.e., a sig feature set for each path data point?
        for i in range(len(self.linear)-1):
            '''
            if i==0:
                x = self.dropout(torch.sigmoid(self.linear[i](x)))
            else:
                x = self.dropout(torch.sigmoid(self.linear[i](x)+x))
            '''
            x = self.dropout(torch.sigmoid(self.linear[i](x)))
        z = F.log_softmax(self.linear[-1](x), dim=1)
        return z
    
    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
                    
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
                    

class CNNSigNLP(nn.Module):
    def __init__(self, vocab_size, out_dimension, pad_idx, c, layers, sig_depth, dropout=0.2, h=1, out_channels=None, stride=None, include_time=True):
        """
        input:
            
        """
        super(CNNSigNLP, self).__init__()
        if not stride:
            stride = (h,c) # non-overlap convolution
        if not out_channels:
            out_channels = c
        in_channels = 100
        self.embed = nn.Embedding(vocab_size, in_channels, pad_idx)
        
        self.conv2d1 = torch.nn.Conv2d(1, out_channels, (h, c), stride=stride)    
        
        self.channels = (in_channels-c)//stride[1]+1 # dimension of the low dimensional paths
        
        self.augment = signatory.Augment(in_channels=in_channels, 
                                         layer_sizes=(), 
                                         kernel_size=1,
                                         include_time=include_time) # include additional time dimension
        sig_channels = signatory.signature_channels(self.channels+include_time,
                                                    depth=sig_depth)
        self.signature = signatory.Signature(depth=sig_depth)
        self.v2 = View([-1, out_channels*sig_channels])
        self.linear = nn.ModuleList([nn.Linear(out_channels*sig_channels, layers[0])])
        for i in range(len(layers)-1):
            self.linear.append(nn.Linear(layers[i], layers[i+1]))
        self.linear.append(nn.Linear(layers[len(layers)-1], out_dimension))
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, text):
        """
        input:
            x - tensor, sequential data
        """
        #x = self.BatchNorm(x)
        
        x = self.dropout(self.embed(text))
        length, batch, _ = x.size()
        x = x.permute(1, 0, 2)
        x = x.reshape(batch, 1, length, -1)
        x = torch.sigmoid(self.conv2d1(x))
        
        T = x.size()[2]
        x = x.view(-1, T, self.channels)
        x = self.augment(x) # augment with time dimension
        x = self.signature(x, basepoint=True)
        #x = self.v2(x)
        x = self.dropout(self.v2(x))
        # question: does the v2 will trace back to the correct structure, i.e., a sig feature set for each path data point?
        for i in range(len(self.linear)-1):
            #x = self.dropout(torch.relu(self.linear[i](x)))
            x = torch.relu(self.linear[i](x))
            
        z = self.linear[-1](x)
        return z
    

class CNNSigNLP_(nn.Module):
    def __init__(self, vocab_size, out_dimension, pad_idx, c, layers, sig_depth, dropout=0.2, h=1, 
                 out_channels=None, stride=None, 
                 include_time=True, bidirectional=True):
        """
        input:
            
        """
        super(CNNSigNLP, self).__init__()
        if not stride:
            stride = (h,c) # non-overlap convolution
        if not out_channels:
            out_channels = c
        in_channels = 100
        num_hidden = 128
        self.embed = nn.Embedding(vocab_size, in_channels, pad_idx)
        
        self.rnn = nn.LSTM(in_channels, 
                           num_hidden, 
                           num_layers=2, 
                           bidirectional=bidirectional, 
                           dropout=dropout)
        
        self.conv2d1 = torch.nn.Conv2d(1, out_channels, (h, c), stride=stride)    
        
        self.channels = (in_channels-c)//stride[1]+1 # dimension of the low dimensional paths
        
        self.augment = signatory.Augment(in_channels=in_channels, 
                                         layer_sizes=(), 
                                         kernel_size=1,
                                         include_time=include_time) # include additional time dimension
        sig_channels = signatory.signature_channels(self.channels+include_time,
                                                    depth=sig_depth)
        self.signature = signatory.Signature(depth=sig_depth)
        self.v2 = View([-1, out_channels*sig_channels])
        self.linear = nn.ModuleList([nn.Linear(num_hidden*2 + out_channels*sig_channels, layers[0])])
        for i in range(len(layers)-1):
            self.linear.append(nn.Linear(layers[i], layers[i+1]))
        self.linear.append(nn.Linear(layers[len(layers)-1], out_dimension))
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, text):
        """
        input:
            x - tensor, sequential data
        """
        #x = self.BatchNorm(x)
        x = self.dropout(self.embed(text))
        _, (hn, cn) = self.rnn(x)
        hn = self.dropout(torch.cat((hn[-2,:,:], hn[-1,:,:]), dim=1))
        
        length, batch, _ = x.size()
        x = x.permute(1, 0, 2)
        x = x.reshape(batch, 1, length, -1)
        x = torch.sigmoid(self.conv2d1(x))
        
        T = x.size()[2]
        x = x.view(-1, T, self.channels)
        x = self.augment(x) # augment with time dimension
        x = self.signature(x, basepoint=True)
        x = self.v2(x)
        
        # combine rnn and cnnsig together
        x = torch.cat((hn, x), dim=1)
        # question: does the v2 will trace back to the correct structure, i.e., a sig feature set for each path data point?
        for i in range(len(self.linear)-1):
            x = self.dropout(torch.relu(self.linear[i](x)))
            
        z = self.linear[-1](x)
        return z




class RNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, n_layers, 
                 bidirectional, dropout, pad_idx):
        
        super(RNN, self).__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx = pad_idx)
        
        self.rnn = nn.LSTM(embedding_dim, 
                           hidden_dim, 
                           num_layers=n_layers, 
                           bidirectional=bidirectional, 
                           dropout=dropout)
        
        self.fc = nn.Linear(hidden_dim * 2, output_dim)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, text, text_lengths):
        
        #text = [sent len, batch size]
        
        embedded = self.dropout(self.embedding(text))
        
        #embedded = [sent len, batch size, emb dim]
        
        #pack sequence
        # lengths need to be on CPU!
        packed_embedded = nn.utils.rnn.pack_padded_sequence(embedded, text_lengths.to('cpu'))
        
        packed_output, (hidden, cell) = self.rnn(packed_embedded)
        
        #unpack sequence
        output, output_lengths = nn.utils.rnn.pad_packed_sequence(packed_output)
        
        hidden = self.dropout(torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim = 1))
                
        #hidden = [batch size, hid dim * num directions]
            
        return self.fc(hidden)