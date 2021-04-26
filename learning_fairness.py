# -*- coding: utf-8 -*-
"""
Created on Tue Nov 10 02:36:48 2020

@author: islam
"""

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#%% Stochastic learning of count model

# compute corresponding counts for each intersection group in a mini-batch
def computeBatchCounts(protectedAttributes,intersectGroups,predictions):
    # intersectGroups should be pre-defined so that stochastic update of p(y|S) 
    # can be maintained correctly among different batches   
     
    # compute counts for each intersectional group
    countsClassOne = torch.zeros((len(intersectGroups)),dtype=torch.float,device=device)
    countsTotal = torch.zeros((len(intersectGroups)),dtype=torch.float,device=device)
    for i in range(len(predictions)):
        index=np.where((intersectGroups==protectedAttributes[i]).all(axis=1))[0][0]
        countsTotal[index] = countsTotal[index] + 1
        countsClassOne[index] = countsClassOne[index] + predictions[i]        
    return countsClassOne, countsTotal

# update count model
class stochasticCountModel(nn.Module):
    # Source: directly taken from deepSurv implementation
    def __init__(self,no_of_groups,N,batch_size):
        super(stochasticCountModel, self).__init__()
        self.countClass_hat = torch.ones((no_of_groups),device=device)
        self.countTotal_hat = torch.ones((no_of_groups),device=device)
        
        self.countClass_hat = self.countClass_hat*(N/(batch_size*no_of_groups)) 
        self.countTotal_hat = self.countTotal_hat*(N/batch_size) 
        
    def forward(self,rho,countClass_batch,countTotal_batch,N,batch_size):
        self.countClass_hat = (1-rho)*self.countClass_hat + rho*(N/batch_size)*countClass_batch
        self.countTotal_hat = (1-rho)*self.countTotal_hat + rho*(N/batch_size)*countTotal_batch

#%%
# \epsilon-DF measurement to form DF-based fairness penalty
def df_train(probabilitiesOfPositive):
    # input: probabilitiesOfPositive = positive p(y|S) from ML algorithm
    # output: epsilon = differential fairness measure
    epsilonPerGroup = torch.zeros(len(probabilitiesOfPositive),dtype=torch.float,device=device)
    for i in  range(len(probabilitiesOfPositive)):
        epsilon = torch.tensor(0.0,device=device) # initialization of DF
        for j in range(len(probabilitiesOfPositive)):
            if i == j:
                continue
            else:
                epsilon = torch.max(epsilon,torch.abs(torch.log(probabilitiesOfPositive[i])-torch.log(probabilitiesOfPositive[j]))) # ratio of probabilities of positive outcome
                epsilon = torch.max(epsilon,torch.abs((torch.log(1-probabilitiesOfPositive[i]))-(torch.log(1-probabilitiesOfPositive[j])))) # ratio of probabilities of negative outcome
        epsilonPerGroup[i] = epsilon # DF per group
    epsilon = torch.max(epsilonPerGroup) # overall DF of the algorithm 
    return epsilon

#%% \epsilon-DF fairness penalty
def df_loss(base_fairness,stochasticModel):
    numClasses = torch.tensor(2.0,device=device)
    concentrationParameter = torch.tensor(1.0,device=device)
    dirichletAlpha = concentrationParameter/numClasses
    zeroTerm = torch.tensor(0.0,device=device) 
    
    theta = (stochasticModel.countClass_hat + dirichletAlpha) /(stochasticModel.countTotal_hat + concentrationParameter)
    #theta = theta/sum(theta)
    epsilonClass = df_train(theta)
    return torch.max(zeroTerm, (epsilonClass-base_fairness))

#%%
# \gamma-SF measurement to form SF-based fairness penalty
def sf_train(probabilitiesOfPositive,alphaSP):
    # input: probabilitiesOfPositive = Pr[D(X)=1|g(x)=1]
    #        alphaG = Pr[g(x)=1]
    # output: gamma-unfairness
    spD = sum(probabilitiesOfPositive*alphaSP) # probabilities of positive class across whole population SP(D) = Pr[D(X)=1]
    gammaPerGroup = torch.zeros(len(probabilitiesOfPositive),dtype=torch.float,device=device) # SF per group
    for i in range(len(probabilitiesOfPositive)):
        gammaPerGroup[i] = alphaSP[i]*torch.abs(spD-probabilitiesOfPositive[i])
    gamma = torch.max(gammaPerGroup) # overall SF of the algorithm 
    return gamma

#%% \epsilon-DF fairness penalty
def sf_loss(base_fairness,stochasticModel):
    numClasses = torch.tensor(2.0,device=device)
    concentrationParameter = torch.tensor(1.0,device=device)
    dirichletAlpha = concentrationParameter/numClasses
    
    zeroTerm = torch.tensor(0.0,device=device) 
    population = sum(stochasticModel.countTotal_hat).detach()
    
    theta = (stochasticModel.countClass_hat + dirichletAlpha) /(stochasticModel.countTotal_hat + concentrationParameter)
    alpha = (stochasticModel.countTotal_hat + dirichletAlpha) /(population + concentrationParameter)
    #theta = theta/sum(theta)
    gammaClass = sf_train(theta,alpha)
    return torch.max(zeroTerm, (gammaClass-base_fairness))