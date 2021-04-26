# -*- coding: utf-8 -*-
"""
Created on Tue Nov 10 01:37:48 2020

@author: islam
"""


import torch
import torch.nn as nn


#%% fully connected deep neural network as binary classifier
class dnnBinary(nn.Module):
    def __init__(self,input_size,hidden,d_prob,acts):
        super(dnnBinary,self).__init__()
        self.fc1 = nn.Linear(input_size,hidden[0])
        self.fc2 = nn.Linear(hidden[0],hidden[1])
        self.fc3 = nn.Linear(hidden[1],hidden[2])
        self.fc4 = nn.Linear(hidden[2],1)
        
        self.d_out = nn.Dropout(d_prob)
        self.out_act = nn.Sigmoid()

        if acts == 'rectify':
            self.act  = nn.ReLU()
        elif acts == 'selu':
            self.act = nn.SELU()
        elif acts == 'elu':
            self.act = nn.ELU()
        elif acts == 'leaky':
            self.act = nn.LeakyReLU()
        elif acts == 'softplus':
            self.act = nn.Softplus()
        
    def forward(self,x):
        out = self.d_out(self.act(self.fc1(x)))
        out = self.d_out(self.act(self.fc2(out)))
        out = self.d_out(self.act(self.fc3(out)))
        out = self.out_act(self.fc4(out))
        return out

#%% Adversarial network on the output to the classifier

class adversaryNet(nn.Module):
    def __init__(self,hidden,d_prob,acts,n_sensitive):
        super(adversaryNet,self).__init__()
        self.fc1 = nn.Linear(1,hidden[0])
        self.fc2 = nn.Linear(hidden[0],hidden[1])
        self.fc3 = nn.Linear(hidden[1],hidden[2])
        self.fc4 = nn.Linear(hidden[2],n_sensitive)
        
        self.d_out = nn.Dropout(d_prob)
        self.out_act = nn.Sigmoid()

        if acts == 'rectify':
            self.act  = nn.ReLU()
        elif acts == 'selu':
            self.act = nn.SELU()
        elif acts == 'elu':
            self.act = nn.ELU()
        elif acts == 'leaky':
            self.act = nn.LeakyReLU()
        elif acts == 'softplus':
            self.act = nn.Softplus()
        
    def forward(self,x):
        out = self.d_out(self.act(self.fc1(x)))
        out = self.d_out(self.act(self.fc2(out)))
        out = self.d_out(self.act(self.fc3(out)))
        out = self.out_act(self.fc4(out))
        return out

