# -*- coding: utf-8 -*-
"""
Created on Tue Nov 10 03:09:04 2020

@author: islam
"""
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from model import dnnBinary, adversaryNet
from learning_fairness import stochasticCountModel, computeBatchCounts, df_loss, sf_loss
from evaluations import computeEDFforData

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#%% train the typical model without fairness constraint
def training_typicalModel(input_size,hiddens,activations,dropOut,learning_rate,num_epochs,trainData,trainLabel,miniBatch,wd):
    typicalModel = dnnBinary(input_size,hiddens,dropOut,activations).to(device)
    # Loss and optimizer
    criterion = nn.BCELoss()
    optimizer = optim.Adam(typicalModel.parameters(),lr = learning_rate,weight_decay=wd)

    # Train the netwok
    typicalModel.train()
    for epoch in range(num_epochs):
        for batch in range(0,np.int64(np.floor(len(trainData)/miniBatch))*miniBatch,miniBatch):
            trainY_batch = trainLabel[batch:(batch+miniBatch)]
            trainX_batch = trainData[batch:(batch+miniBatch)]
            
            trainX_batch = Variable(trainX_batch.float()).to(device)
            trainY_batch = Variable(trainY_batch.float()).to(device)
            
            # forward + backward + optimize
            outputs = typicalModel(trainX_batch)
            tot_loss = criterion(outputs, trainY_batch)
            
            # zero the parameter gradients
            optimizer.zero_grad()
            tot_loss.backward()
            optimizer.step() 
    return typicalModel

#%% train the DF model with \epsilon-based fairness constraint
def training_df_model(input_size,hiddens,activations,dropOut,learning_rate,num_epochs,trainData,trainLabel,miniBatch,S,intersectionalGroups,burnin_iters,stepSize,epsilonBase,lamda,wd):
    
    VB_CountModel = stochasticCountModel(len(intersectionalGroups),len(trainData),miniBatch)
    burnCount = 0 # to break burn-in train after 100 mini-batch iterations for large datasets
    
    df_model = dnnBinary(input_size,hiddens,dropOut,activations).to(device)
    # Loss and optimizer
    criterion = nn.BCELoss()
    optimizer = optim.Adam(df_model.parameters(),lr = learning_rate, weight_decay=wd)

    # Train the netwok with df constraint
    df_model.train()
    for batch in range(0,np.int64(np.floor(len(trainData)/miniBatch))*miniBatch,miniBatch):
        trainS_batch = S[batch:(batch+miniBatch)] # protected attributes in the mini-batch
        trainY_batch = trainLabel[batch:(batch+miniBatch)]
        trainX_batch = trainData[batch:(batch+miniBatch)]
        
        trainX_batch = Variable(trainX_batch.float()).to(device)
        trainY_batch = Variable(trainY_batch.float()).to(device)

        # forward + backward + optimize
        outputs = df_model(trainX_batch)
        tot_loss = criterion(outputs, trainY_batch)
        
        # zero the parameter gradients
        optimizer.zero_grad()
        tot_loss.backward()
        optimizer.step()
        
        # count burn in iterations
        if burnCount==burnin_iters:
            break
        burnCount += 1
        
    for epoch in range(num_epochs):
        for batch in range(0,np.int64(np.floor(len(trainData)/miniBatch))*miniBatch,miniBatch):
            trainS_batch = S[batch:(batch+miniBatch)] # protected attributes in the mini-batch
            trainY_batch = trainLabel[batch:(batch+miniBatch)]
            trainX_batch = trainData[batch:(batch+miniBatch)]
            
            trainX_batch = Variable(trainX_batch.float()).to(device)
            trainY_batch = Variable(trainY_batch.float()).to(device)
            
            VB_CountModel.countClass_hat.detach_()
            VB_CountModel.countTotal_hat.detach_()
            # forward + backward + optimize
            outputs = df_model(trainX_batch)
            loss = criterion(outputs, trainY_batch)

            # update Count model 
            countClass, countTotal = computeBatchCounts(trainS_batch,intersectionalGroups,outputs)
            #thetaModel(stepSize,theta_batch)
            VB_CountModel(stepSize,countClass, countTotal,len(trainData),miniBatch)
            
            # fairness constraint 
            epsilon_loss = df_loss(epsilonBase,VB_CountModel)            
            tot_loss = loss+lamda*epsilon_loss
            
            # zero the parameter gradients
            optimizer.zero_grad()
            tot_loss.backward()
            optimizer.step() 
    return df_model

#%% adversarial debiasing
def pretrain_adversary(numSensitive,hiddens,activations,dropOut,learning_rate,num_preTrain_epochs,trainData,trainLabel,miniBatch,wd,S,lamda,clfModel):
    advModel = adversaryNet(hiddens,dropOut,activations,numSensitive).to(device)
    # Loss and optimizer
    adv_criterion = nn.BCELoss(reduce=False)
    adv_optimizer = optim.Adam(advModel.parameters(),lr = learning_rate,weight_decay=wd)

    # Train the netwok
    advModel.train()
    for epoch in range(num_preTrain_epochs):
        for batch in range(0,np.int64(np.floor(len(trainData)/miniBatch))*miniBatch,miniBatch):
            trainS_batch = S[batch:(batch+miniBatch)]
            trainX_batch = trainData[batch:(batch+miniBatch)]
            
            trainX_batch = Variable(trainX_batch.float()).to(device)
            trainS_batch = Variable(trainS_batch.float()).to(device)
            
            # forward + backward + optimize
            p_y = clfModel(trainX_batch).detach()
            p_z  = advModel(p_y)
            
            loss = (adv_criterion(p_z, trainS_batch)*lamda).mean()
            
            # zero the parameter gradients
            adv_optimizer.zero_grad()
            loss.backward()
            adv_optimizer.step() 
    return advModel

def training_adversarial_debiasing(input_size,hiddens,activations,dropOut,learning_rate,num_preTrain_epochs,num_epochs,trainData,trainLabel,miniBatch,wd,S,lamda):
    # pretrain typical classifier
    clfModel = training_typicalModel(input_size,hiddens,activations,dropOut,learning_rate,num_preTrain_epochs,trainData,trainLabel,miniBatch,wd)
    # pretrain adversarial network
    numSensitive = S.size()[1]
    advModel = pretrain_adversary(numSensitive,hiddens,activations,dropOut,learning_rate,num_preTrain_epochs,trainData,trainLabel,miniBatch,wd,S,lamda,clfModel)
    
    # Loss and optimizer
    clf_criterion = nn.BCELoss()
    clf_optimizer = optim.Adam(clfModel.parameters(),lr = learning_rate, weight_decay=wd)
    
    adv_criterion = nn.BCELoss(reduce=False)
    adv_optimizer = optim.Adam(advModel.parameters(),lr = learning_rate,weight_decay=wd)

    # Train both networks
    for epoch in range(num_epochs):
        # Train classifier on single batch
        for batch in range(0,np.int64(np.floor(len(trainData)/miniBatch))*miniBatch,miniBatch):            
            # Train adversary on full dataset
            for adv_batch in range(0,np.int64(np.floor(len(trainData)/miniBatch))*miniBatch,miniBatch):
                trainS_adv = S[adv_batch:(adv_batch+miniBatch)]
                trainX_adv = trainData[adv_batch:(adv_batch+miniBatch)]
                
                trainX_adv = Variable(trainX_adv.float()).to(device)
                trainS_adv = Variable(trainS_adv.float()).to(device)
                
                # forward + backward + optimize
                p_y = clfModel(trainX_adv)
                p_z  = advModel(p_y)
                
                adv_loss = (adv_criterion(p_z, trainS_adv)*lamda).mean()
                
                # zero the parameter gradients
                adv_optimizer.zero_grad()
                adv_loss.backward()
                adv_optimizer.step() 

            # Train classifier on single batch
            # forward + backward + optimize
            trainY_batch = trainLabel[batch:(batch+miniBatch)]
            trainX_batch = trainData[batch:(batch+miniBatch)]
            trainS_batch = S[batch:(batch+miniBatch)]
            
            trainX_batch = Variable(trainX_batch.float()).to(device)
            trainY_batch = Variable(trainY_batch.float()).to(device)
            trainS_batch = Variable(trainS_batch.float()).to(device)
            
            p_y = clfModel(trainX_batch)
            p_z  = advModel(p_y)
            
            adv_loss = (adv_criterion(p_z, trainS_batch)*lamda).mean()
            clf_loss = clf_criterion(p_y, trainY_batch) - (adv_criterion(advModel(p_y), trainS_batch)*lamda).mean()
            
            # zero the parameter gradients
            clf_optimizer.zero_grad()
            clf_loss.backward()
            clf_optimizer.step() 
    return clfModel

#%% load pre-trained typical model
def load_preTrainedModel(input_size,hiddens,dropOut,activations, path):
    typ_model = dnnBinary(input_size,hiddens,dropOut,activations).to(device)
    typ_model.load_state_dict(torch.load(path))
    typ_model.to(device)
    return typ_model