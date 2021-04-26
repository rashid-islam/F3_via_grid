# -*- coding: utf-8 -*-
"""
Created on Tue Nov 10 02:48:35 2020

@author: islam
"""

import pandas as pd
import numpy as np
#%%
# Measure \epsilon-DF from positive predict probabilities
def differentialFairnessBinaryOutcome(probabilitiesOfPositive):
    # input: probabilitiesOfPositive = positive p(y|S) from ML algorithm
    # output: epsilon = differential fairness measure
    epsilonPerGroup = np.zeros(len(probabilitiesOfPositive))
    for i in  range(len(probabilitiesOfPositive)):
        epsilon = 0.0 # initialization of DF
        for j in range(len(probabilitiesOfPositive)):
            if i == j:
                continue
            else:
                epsilon = max(epsilon,abs(np.log(probabilitiesOfPositive[i])-np.log(probabilitiesOfPositive[j]))) # ratio of probabilities of positive outcome
                epsilon = max(epsilon,abs(np.log((1-probabilitiesOfPositive[i]))-np.log((1-probabilitiesOfPositive[j])))) # ratio of probabilities of negative outcome
        epsilonPerGroup[i] = epsilon # DF per group
    epsilon = max(epsilonPerGroup) # overall DF of the algorithm 
    return epsilon

#%%
# Measure \gamma-SF (gamma unfairness) 
def subgroupFairness(probabilitiesOfPositive,alphaSP):
    # input: probabilitiesOfPositive = Pr[D(X)=1|g(x)=1]
    #        alphaG = Pr[g(x)=1]
    # output: gamma-unfairness
    spD = sum(probabilitiesOfPositive*alphaSP) # probabilities of positive class across whole population SP(D) = Pr[D(X)=1]
    gammaPerGroup = np.zeros(len(probabilitiesOfPositive)) # SF per group
    for i in range(len(probabilitiesOfPositive)):
        gammaPerGroup[i] = alphaSP[i]*abs(spD-probabilitiesOfPositive[i])
    gamma = max(gammaPerGroup) # overall SF of the algorithm 
    return gamma

#%% intersectional fairness measurement from smoothed empirical counts 
def computeEDFforData(protectedAttributes,predictions,predictProb,intersectGroups):
    # compute counts and probabilities
    countsClassOne = np.zeros(len(intersectGroups))
    countsTotal = np.zeros(len(intersectGroups))
    countsClassOne_soft = np.zeros(len(intersectGroups))
    
    numClasses = 2
    concentrationParameter = 1.0
    dirichletAlpha = concentrationParameter/numClasses
    population = len(predictions)
    
    for i in range(len(predictions)):
        index=np.where((intersectGroups==protectedAttributes[i]).all(axis=1))[0][0]
        countsTotal[index] = countsTotal[index] + 1
        countsClassOne_soft[index] = countsClassOne_soft[index] + predictProb[i]
        if predictions[i] == 1:
            countsClassOne[index] = countsClassOne[index] + 1          
    
    # probability of y given S (p(y=1|S)): probability distribution over merit per value of the protected attributes
    probabilitiesOfPositive_hard = (countsClassOne + dirichletAlpha) /(countsTotal + concentrationParameter)
    probabilitiesOfPositive_soft = (countsClassOne_soft + dirichletAlpha) /(countsTotal + concentrationParameter)
    alphaG_smoothed = (countsTotal + dirichletAlpha) /(population + concentrationParameter)

    epsilon_hard = differentialFairnessBinaryOutcome(probabilitiesOfPositive_hard)
    gamma_hard = subgroupFairness(probabilitiesOfPositive_hard,alphaG_smoothed)
    
    epsilon_soft = differentialFairnessBinaryOutcome(probabilitiesOfPositive_soft)
    gamma_soft = subgroupFairness(probabilitiesOfPositive_soft,alphaG_smoothed)
    
    return epsilon_hard,epsilon_soft,gamma_hard,gamma_soft

#%% dempographic parity (dp) requires the probability for an individual to be assigned
# the favourable outcome to be equal across the privileged and unprivileged groups.

def dp_distance(binaryGroup, class_labels,predictProb):

    """ It is often impossible or undesirable to satisfy demographic parity exactly 
        (i.e. achieve complete independence).
        In this case, a useful metric is demographic parity distance """

    non_prot_all = sum(binaryGroup == 1) # privileged group
    prot_all = sum(binaryGroup == 0) # unprivileged group
    
    # smoothing parameter
    numClasses = 2
    concentrationParameter = 1.0
    dirichletAlpha = concentrationParameter/numClasses
    
    non_prot_pos = sum(class_labels[binaryGroup == 1] == 1) # privileged in positive class
    prot_pos = sum(class_labels[binaryGroup == 0] == 1) # unprivileged in positive class
    frac_non_prot_pos = float(non_prot_pos+dirichletAlpha) / float(non_prot_all+concentrationParameter)
    frac_prot_pos = float(prot_pos+dirichletAlpha) / float(prot_all+concentrationParameter)
    dp_hard = abs(frac_prot_pos-frac_non_prot_pos)
    
    # soft p-rule
    non_prot_pos_soft = sum(predictProb[binaryGroup == 1]) # privileged in positive class
    prot_pos_soft = sum(predictProb[binaryGroup == 0]) # unprivileged in positive class
    frac_non_prot_pos_soft = float(non_prot_pos_soft+dirichletAlpha) / float(non_prot_all+concentrationParameter)
    frac_prot_pos_soft = float(prot_pos_soft+dirichletAlpha) / float(prot_all+concentrationParameter)
    dp_soft = abs(frac_prot_pos_soft-frac_non_prot_pos_soft)
    
    return dp_hard,dp_soft
#%% p%-rule disparate impact
def p_rule(x_control, class_labels, predictProb):

    """ Compute the p-rule based on Doctrine of disparate impact """

    non_prot_all = sum(x_control == 1) # non-protected group
    prot_all = sum(x_control == 0) # protected group
    
    numClasses = 2
    concentrationParameter = 1.0
    dirichletAlpha = concentrationParameter/numClasses
    
    non_prot_pos = sum(class_labels[x_control == 1] == 1) # non_protected in positive class
    prot_pos = sum(class_labels[x_control == 0] == 1) # protected in positive class
    frac_non_prot_pos = float(non_prot_pos+dirichletAlpha) / float(non_prot_all+concentrationParameter)
    frac_prot_pos = float(prot_pos+dirichletAlpha) / float(prot_all+concentrationParameter)
    p_rule = min(frac_prot_pos / frac_non_prot_pos, frac_non_prot_pos / frac_prot_pos) * 100.0
    
    # soft p-rule
    non_prot_pos_soft = sum(predictProb[x_control == 1]) # non_protected in positive class
    prot_pos_soft = sum(predictProb[x_control == 0]) # protected in positive class
    frac_non_prot_pos_soft = float(non_prot_pos_soft+dirichletAlpha) / float(non_prot_all+concentrationParameter)
    frac_prot_pos_soft = float(prot_pos_soft+dirichletAlpha) / float(prot_all+concentrationParameter)
    p_rule_soft = min(frac_prot_pos_soft / frac_non_prot_pos_soft,frac_non_prot_pos_soft / frac_prot_pos_soft) * 100.0
    
    return p_rule,p_rule_soft


#%% Evaluate model
import torch
import torch.nn as nn
from torch.autograd import Variable
from sklearn.metrics import accuracy_score, roc_auc_score,f1_score

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def evaluate_model(model, data, S, y, intersects):
    model.eval()
    with torch.no_grad():
        data = Variable(data.float()).to(device)
        predictProb = model(data)
        if predictProb.device.type == 'cuda':
            predictProb = predictProb.cpu()
        predicted = ((predictProb>0.5).numpy()).reshape((-1,))
        Accuracy = sum(predicted == y)/len(y)
        
    predictProb = (predictProb.numpy()).reshape((-1,))
    
    aucScore = roc_auc_score(y,predictProb)
    nn_f1 = f1_score(y,predicted)
    
    epsilon_hard,epsilon_soft,gamma_hard,gamma_soft = computeEDFforData(S,predicted,predictProb,intersects)
    return predictProb, predicted, Accuracy, aucScore, nn_f1, epsilon_hard, epsilon_soft, gamma_hard, gamma_soft
