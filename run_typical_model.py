# -*- coding: utf-8 -*-
"""
Created on Tue Nov 10 03:39:47 2020

@author: islam
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import sklearn as sk
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score,f1_score

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

from utilities import load_compas_data, dp_group_compas
from evaluations import evaluate_model, dp_distance, p_rule
from train_models import training_typicalModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#The function below ensures that we seed all random generators with the same value to get reproducible results
def set_random_seed(state=1):
    gens = (np.random.seed, torch.manual_seed, torch.cuda.manual_seed)
    for set_state in gens:
        set_state(state)

RANDOM_STATE = 1
set_random_seed(RANDOM_STATE)

#%% data loading and preprocessing
# load the train dataset
X, y, S = load_compas_data('data/compas-scores-two-years.csv')

# Define all the "intersectional groups" to maintain stochastic update of p(y|S) correctly among different batches 
intersectionalGroups = np.unique(S,axis=0) # all intersecting groups, i.e. black-women, white-man etc  

#%%
# data pre-processing
# scale/normalize train & test data and shuffle train data
X, test_X, y, test_y, S, test_S = train_test_split(X, y, S, test_size=0.20, 
                                                                     stratify=S, random_state=7)

scaler = StandardScaler().fit(X)
scale_df = lambda df, scaler: pd.DataFrame(scaler.transform(df), columns=df.columns, index=df.index)
X = X.pipe(scale_df, scaler) 
test_X = test_X.pipe(scale_df, scaler)


X, y, S = sk.utils.shuffle(X, y, S, random_state=0)

X = X.values 
y = y.values 

test_X = test_X.values 
test_y = test_y.values 


X, dev_X, y, dev_y, S, dev_S = train_test_split(X, y, S, test_size=0.20,stratify=S, random_state=7)


#%%
# deep neural network using pytorch
trainData = torch.from_numpy(X)
trainLabel = torch.from_numpy(y.reshape((-1,1)))

#devData = torch.from_numpy(devData)                       
testData = torch.from_numpy(test_X)
devData = torch.from_numpy(dev_X)

#%% demographic parity specific parameters
# will simplify later
dp_race_train,dp_gender_train,dp_most_under_train,dp_race_dev,dp_gender_dev,dp_most_under_dev,\
dp_race_test,dp_gender_test,dp_most_under_test = dp_group_compas(S, dev_S, test_S)

#%% hyperparameters grid
input_size = trainData.size()[1]
num_epochs = 10

hidden_layers = [[64, 32, 16], [64, 64, 64], [32, 32, 32]]
mini_batch_sizes = [128, 256] # mini-batch size
learning_rates = [0.001, 0.005]
drop_probs = [0.0, 0.5]
activations = ['rectify', 'leaky']
weight_decays = [0.0, 1e-5]

numHypers = len(hidden_layers)*len(mini_batch_sizes)*len(learning_rates)*len(drop_probs)*len(activations)*len(weight_decays)

best_net = None # store the best model into this
best_acc = -1

#%% Report for experimental results
# result columns: accuracy, auc score, f1 score, epsilon_hard, epsilon_soft, gamma_hard, gamma_soft, dp_race_hard, dp_race_soft,dp_gender_hard, dp_gender_soft, dp_nation_hard, dp_nation_soft, dp_most_under_hard, dp_most_under_soft
coulumn_names = ['accuracy', 'auc score', 'f1 score', 'DF-hard', 'DF-soft', 'SF-hard', 'SF-soft', 'DP-race-hard', 'DP-race-soft','DP-gender-hard', 'DP-gender-soft', 'DP-most-under-hard', 'DP-most-under-soft', \
                 'P-race-hard', 'P-race-soft','P-gender-hard', 'P-gender-soft', 'P-most-under-hard', 'P-most-under-soft']
Train_results = np.zeros((numHypers,len(coulumn_names)))
Train_results = pd.DataFrame(Train_results, columns=coulumn_names)

Dev_results = np.zeros((numHypers,len(coulumn_names)))
Dev_results = pd.DataFrame(Dev_results, columns=coulumn_names)

Test_results = np.zeros((numHypers,len(coulumn_names)))
Test_results = pd.DataFrame(Test_results, columns=coulumn_names)

#%% train typical model by grid search over hyperparameters
import sys
sys.stdout=open("results/typical-model/typical_model_report.txt","w")

print(f"Training typical models by grid search over hyperparameters")
print('\n')
print('\n')
run = 0 # to track n-th model
for hiddens in hidden_layers:
    for lr in learning_rates: 
        for d_out in drop_probs:
                for a_func in activations:
                    for miniBatch in mini_batch_sizes:
                        for wd in weight_decays:
                            print(f"Model: {run: d}")
                            
                            set_random_seed(RANDOM_STATE)
                            
                            typ_Model = training_typicalModel(input_size,hiddens,a_func,d_out,lr,num_epochs,trainData,trainLabel,miniBatch,wd)
                            print(f"Trained with the following hyperparameters:")
                            print('hidden layers (nodes/layer): ', hiddens, 'learning rate: ', lr, 'Drop out prob: ', d_out, 'mini-batch sizes: ', miniBatch, 'activation function: ', a_func, 'weight decay: ', wd)
                            print('\n')
                            
                            # Evaluate on training set
                            pred_prob_train, pred_label_train, Train_results['accuracy'][run], Train_results['auc score'][run], Train_results['f1 score'][run], Train_results['DF-hard'][run], Train_results['DF-soft'][run],\
                            Train_results['SF-hard'][run], Train_results['SF-soft'][run] = evaluate_model(typ_Model, trainData, S, trainLabel.numpy().reshape((-1,)), intersectionalGroups)
                            
                            Train_results['DP-race-hard'][run],Train_results['DP-race-soft'][run] = dp_distance(dp_race_train, pred_label_train, pred_prob_train)
                            Train_results['DP-gender-hard'][run],Train_results['DP-gender-soft'][run] = dp_distance(dp_gender_train, pred_label_train, pred_prob_train)
                            Train_results['DP-most-under-hard'][run],Train_results['DP-most-under-soft'][run] = dp_distance(dp_most_under_train, pred_label_train, pred_prob_train)

                            Train_results['P-race-hard'][run],Train_results['P-race-soft'][run] = p_rule(dp_race_train, pred_label_train, pred_prob_train)
                            Train_results['P-gender-hard'][run],Train_results['P-gender-soft'][run] = p_rule(dp_gender_train, pred_label_train, pred_prob_train)
                            Train_results['P-most-under-hard'][run],Train_results['P-most-under-soft'][run] = p_rule(dp_most_under_train, pred_label_train, pred_prob_train)

                            # Evaluate on dev set
                            pred_prob_dev, pred_label_dev, Dev_results['accuracy'][run], Dev_results['auc score'][run], Dev_results['f1 score'][run], Dev_results['DF-hard'][run], Dev_results['DF-soft'][run],\
                            Dev_results['SF-hard'][run], Dev_results['SF-soft'][run] = evaluate_model(typ_Model, devData, dev_S, dev_y, intersectionalGroups)
                            
                            Dev_results['DP-race-hard'][run],Dev_results['DP-race-soft'][run] = dp_distance(dp_race_dev, pred_label_dev, pred_prob_dev)
                            Dev_results['DP-gender-hard'][run],Dev_results['DP-gender-soft'][run] = dp_distance(dp_gender_dev, pred_label_dev, pred_prob_dev)
                            Dev_results['DP-most-under-hard'][run],Dev_results['DP-most-under-soft'][run] = dp_distance(dp_most_under_dev, pred_label_dev, pred_prob_dev)
                            
                            Dev_results['P-race-hard'][run],Dev_results['P-race-soft'][run] = p_rule(dp_race_dev, pred_label_dev, pred_prob_dev)
                            Dev_results['P-gender-hard'][run],Dev_results['P-gender-soft'][run] = p_rule(dp_gender_dev, pred_label_dev, pred_prob_dev)
                            Dev_results['P-most-under-hard'][run],Dev_results['P-most-under-soft'][run] = p_rule(dp_most_under_dev, pred_label_dev, pred_prob_dev)

                            # Evaluate on test set
                            pred_prob_test, pred_label_test, Test_results['accuracy'][run], Test_results['auc score'][run], Test_results['f1 score'][run], Test_results['DF-hard'][run], Test_results['DF-soft'][run],\
                            Test_results['SF-hard'][run], Test_results['SF-soft'][run] = evaluate_model(typ_Model, testData, test_S, test_y, intersectionalGroups)
                            
                            Test_results['DP-race-hard'][run],Test_results['DP-race-soft'][run] = dp_distance(dp_race_test, pred_label_test, pred_prob_test)
                            Test_results['DP-gender-hard'][run],Test_results['DP-gender-soft'][run] = dp_distance(dp_gender_test, pred_label_test, pred_prob_test)
                            Test_results['DP-most-under-hard'][run],Test_results['DP-most-under-soft'][run] = dp_distance(dp_most_under_test, pred_label_test, pred_prob_test)
                            
                            Test_results['P-race-hard'][run],Test_results['P-race-soft'][run] = p_rule(dp_race_test, pred_label_test, pred_prob_test)
                            Test_results['P-gender-hard'][run],Test_results['P-gender-soft'][run] = p_rule(dp_gender_test, pred_label_test, pred_prob_test)
                            Test_results['P-most-under-hard'][run],Test_results['P-most-under-soft'][run] = p_rule(dp_most_under_test, pred_label_test, pred_prob_test)

                            # choose the best model based on a pre-defined condition
                            if Dev_results['accuracy'][run]>best_acc:
                                best_acc = Dev_results['accuracy'][run]
                                print('\n')
                                print(f"Current best model:")
                                print('hidden layers (nodes/layer): ', hiddens, 'learning rate: ', lr, 'Drop out prob: ', d_out, 'mini-batch sizes: ', miniBatch, 'activation function: ', a_func, 'weight decay: ', wd)
                                print(f"accuracy on dev set: {Dev_results['accuracy'][run]: .4f}")
                                best_net = typ_Model
                                print('\n')
                                print('\n')
                            
                            run += 1

#%% Final evaluation on the best model
                        
# Evaluate on training set
pred_prob_train, pred_label_train, acc, auc, f1, df_hard, df_soft,\
sf_hard,sf_soft = evaluate_model(best_net, trainData, S, trainLabel.numpy().reshape((-1,)), intersectionalGroups)

dp_race_hard,dp_race_soft = dp_distance(dp_race_train, pred_label_train, pred_prob_train)
dp_gender_hard,dp_gender_soft = dp_distance(dp_gender_train, pred_label_train, pred_prob_train)
dp_most_under_hard,dp_most_under_soft = dp_distance(dp_most_under_train, pred_label_train, pred_prob_train)

p_race_hard,p_race_soft = p_rule(dp_race_train, pred_label_train, pred_prob_train)
p_gender_hard,p_gender_soft = p_rule(dp_gender_train, pred_label_train, pred_prob_train)
p_most_under_hard,p_most_under_soft = p_rule(dp_most_under_train, pred_label_train, pred_prob_train)

print('\n')
print('For the selected best model,')
print('\n')
print('Evaluation results on the training set:')
print(f"accuracy: {acc: .3f}")
print(f"auc: {auc: .3f}")
print(f"f1 score: {f1: .3f}")

print(f"df_hard: {df_hard: .3f}")
print(f"df_soft: {df_soft: .3f}")
print(f"sf_hard: {sf_hard: .3f}")
print(f"sf_soft: {sf_soft: .3f}")

print(f"dp_race_hard: {dp_race_hard: .3f}")
print(f"dp_race_soft: {dp_race_soft: .3f}")
print(f"dp_gender_hard: {dp_gender_hard: .3f}")
print(f"dp_gender_soft: {dp_gender_soft: .3f}")
print(f"dp_most_under_hard: {dp_most_under_hard: .3f}")
print(f"dp_most_under_soft: {dp_most_under_soft: .3f}")

print(f"p_race_hard: {p_race_hard: .3f}")
print(f"p_race_soft: {p_race_soft: .3f}")
print(f"p_gender_hard: {p_gender_hard: .3f}")
print(f"p_gender_soft: {p_gender_soft: .3f}")
print(f"p_most_under_hard: {p_most_under_hard: .3f}")
print(f"p_most_under_soft: {p_most_under_soft: .3f}")
# Evaluate on dev set
pred_prob_dev, pred_label_dev, acc, auc, f1, df_hard, df_soft,\
sf_hard,sf_soft = evaluate_model(best_net, devData, dev_S, dev_y, intersectionalGroups)

dp_race_hard,dp_race_soft = dp_distance(dp_race_dev, pred_label_dev, pred_prob_dev)
dp_gender_hard,dp_gender_soft = dp_distance(dp_gender_dev, pred_label_dev, pred_prob_dev)
dp_most_under_hard,dp_most_under_soft = dp_distance(dp_most_under_dev, pred_label_dev, pred_prob_dev)

p_race_hard,p_race_soft = p_rule(dp_race_dev, pred_label_dev, pred_prob_dev)
p_gender_hard,p_gender_soft = p_rule(dp_gender_dev, pred_label_dev, pred_prob_dev)
p_most_under_hard,p_most_under_soft = p_rule(dp_most_under_dev, pred_label_dev, pred_prob_dev)

print('\n')
print('Evaluation results on the developement set:')
print(f"accuracy: {acc: .3f}")
print(f"auc: {auc: .3f}")
print(f"f1 score: {f1: .3f}")

print(f"df_hard: {df_hard: .3f}")
print(f"df_soft: {df_soft: .3f}")
print(f"sf_hard: {sf_hard: .3f}")
print(f"sf_soft: {sf_soft: .3f}")

print(f"dp_race_hard: {dp_race_hard: .3f}")
print(f"dp_race_soft: {dp_race_soft: .3f}")
print(f"dp_gender_hard: {dp_gender_hard: .3f}")
print(f"dp_gender_soft: {dp_gender_soft: .3f}")
print(f"dp_most_under_hard: {dp_most_under_hard: .3f}")
print(f"dp_most_under_soft: {dp_most_under_soft: .3f}")

print(f"p_race_hard: {p_race_hard: .3f}")
print(f"p_race_soft: {p_race_soft: .3f}")
print(f"p_gender_hard: {p_gender_hard: .3f}")
print(f"p_gender_soft: {p_gender_soft: .3f}")
print(f"p_most_under_hard: {p_most_under_hard: .3f}")
print(f"p_most_under_soft: {p_most_under_soft: .3f}")
# Evaluate on test set
pred_prob_test, pred_label_test, acc, auc, f1, df_hard, df_soft,\
sf_hard,sf_soft = evaluate_model(best_net, testData, test_S, test_y, intersectionalGroups)

dp_race_hard,dp_race_soft = dp_distance(dp_race_test, pred_label_test, pred_prob_test)
dp_gender_hard,dp_gender_soft = dp_distance(dp_gender_test, pred_label_test, pred_prob_test)
dp_most_under_hard,dp_most_under_soft = dp_distance(dp_most_under_test, pred_label_test, pred_prob_test)

p_race_hard,p_race_soft = p_rule(dp_race_test, pred_label_test, pred_prob_test)
p_gender_hard,p_gender_soft = p_rule(dp_gender_test, pred_label_test, pred_prob_test)
p_most_under_hard,p_most_under_soft = p_rule(dp_most_under_test, pred_label_test, pred_prob_test)

print('\n')
print('Evaluation results on the test set:')
print(f"accuracy: {acc: .3f}")
print(f"auc: {auc: .3f}")
print(f"f1 score: {f1: .3f}")

print(f"df_hard: {df_hard: .3f}")
print(f"df_soft: {df_soft: .3f}")
print(f"sf_hard: {sf_hard: .3f}")
print(f"sf_soft: {sf_soft: .3f}")

print(f"dp_race_hard: {dp_race_hard: .3f}")
print(f"dp_race_soft: {dp_race_soft: .3f}")
print(f"dp_gender_hard: {dp_gender_hard: .3f}")
print(f"dp_gender_soft: {dp_gender_soft: .3f}")
print(f"dp_most_under_hard: {dp_most_under_hard: .3f}")
print(f"dp_most_under_soft: {dp_most_under_soft: .3f}")

print(f"p_race_hard: {p_race_hard: .3f}")
print(f"p_race_soft: {p_race_soft: .3f}")
print(f"p_gender_hard: {p_gender_hard: .3f}")
print(f"p_gender_soft: {p_gender_soft: .3f}")
print(f"p_most_under_hard: {p_most_under_hard: .3f}")
print(f"p_most_under_soft: {p_most_under_soft: .3f}")

#sys.stdout.close()

#%% saving model and results

torch.save(best_net.state_dict(), "trained-models/TypicalModel")

np.savetxt('results/typical-model/pred_prob_test.txt',pred_prob_test)

Train_results.to_csv('results/typical-model/Train_results.csv',index=False)
Dev_results.to_csv('results/typical-model/Dev_results.csv',index=False)
Test_results.to_csv('results/typical-model/Test_results.csv',index=False)
 
