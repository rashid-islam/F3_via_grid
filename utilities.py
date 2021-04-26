# -*- coding: utf-8 -*-
"""
Created on Tue Nov 10 01:26:03 2020

@author: islam
"""

import pandas as pd
import numpy as np


#%% 
# Multiple benchmark datasets for fairness study

#%% Adult data
# income predictions on "Census Income" dataset 
# race, gender & nationality selected as protected attributes
## parse the dataset into three dataset: features (X), targets (y) and protected attributes (S)
def load_census_data (path,check):
    column_names = ['age', 'workclass','fnlwgt','education','education_num',
                    'marital_status','occupation','relationship','race','gender',
                    'capital_gain','capital_loss','hours_per_week','nationality','target']
    input_data = (pd.read_csv(path,names=column_names,
                               na_values="?",sep=r'\s*,\s*',engine='python'))
    # sensitive attributes; we identify 'race','gender' and 'nationality' as sensitive attributes
    # note : keeping the protected attributes in the data set, but make sure they are converted to same category as in the S
    input_data['race'] = input_data['race'].map({'Black': 0,'White': 1, 'Asian-Pac-Islander': 2, 'Amer-Indian-Eskimo': 3, 'Other': 3})
    input_data['gender'] = (input_data['gender'] == 'Male').astype(int)
    input_data['nationality'] = (input_data['nationality'] == 'United-States').astype(int)
    
    protected_attribs = ['race', 'gender','nationality']
    S = (input_data.loc[:, protected_attribs])
   
    # targets; 1 when someone makes over 50k , otherwise 0
    if(check):
        y = (input_data['target'] == '>50K').astype(int)    # target 1 when income>50K
    else:
        y = (input_data['target'] == '>50K.').astype(int)    # target 1 when income>50K
    
    X = (input_data
         .drop(columns=['target'])
         .fillna('Unknown')
         .pipe(pd.get_dummies, drop_first=True))
    return X, y, S

#%% COMPAS data
# parse the dataset into three dataset: features (X), targets (y) and protected attributes (S)
def load_compas_data (path):
    input_data = (pd.read_csv(path))
    protectedFeatures = ['race', 'sex']
    X = (input_data.loc[:, protectedFeatures])
    X['race'] = X['race'].map({'African-American': 0,'Caucasian': 1, 'Hispanic': 2, 'Native American': 3,'Asian': 3, 'Other': 3})
    X['sex'] = X['sex'].map({'Female': 0,'Male': 1})
    y = input_data['two_year_recid']
    predictions = input_data['score_text']
    predictions = predictions.map({'Low': 0,'Medium': 1,'High': 1})
    features = ['sex','age','age_cat','race','juv_fel_count','decile_score','juv_misd_count','juv_other_count','priors_count',
                'days_b_screening_arrest','c_days_from_compas','is_violent_recid']
    dataSet = (input_data.loc[:, features])
    dataSet = (dataSet
         .fillna('Unknown')
         .pipe(pd.get_dummies, drop_first=True))
    return dataSet,y,X.values

#%% Health heritage prize (HHP) data
def load_hhp_data(path):
    data = pd.read_csv(path)
    sex = data['sexMISS'] == 0
    age = data['age_MISS'] == 0
    
    data = data[sex & age]
    data = data.drop(['DaysInHospital','MemberID_t','trainset','sexMISS','age_MISS'], axis=1)
    # Year 3 of 'DaysInHospital' is not complete (all intances are nan), so we drop this column
    data['YEAR_t'] = data['YEAR_t'].map({'Y1': 0,'Y2': 1, 'Y3': 2})
    
    labels = (data['CharlsonIndexI_max']> 0).astype(int) # The downstream task is to predict whether the 
                                                        # Charlson Index (an estimation of patient mortality) 
                                                        # in a given year is greater than zero
    # drop all charlson index related attributes
    data = data.drop(['CharlsonIndexI_max','CharlsonIndexI_min','CharlsonIndexI_ave','CharlsonIndexI_range','CharlsonIndexI_stdev'], axis=1)
    # Number of patients = 171067
    
    ages = (data[['age_%d5' % (i) for i in range(0, 9)]]).values
    sexs = (data[['sexMALE', 'sexFEMALE']]).values
    # ages 85 encoded by 0 and sex female encoded by 0
    
    uniqueAges = np.unique(ages,axis=0)
    uniqueSexs = np.unique(sexs,axis=0)
    
    protected_attr = np.int64(np.zeros((len(data),2)))
    
    # for protected attributes in categorical
    for i in range(len(data)):
        indx_sex=np.where((uniqueSexs==sexs[i]).all(axis=1))[0][0]
        protected_attr[i,0] = indx_sex
        
        indx_age=np.where((uniqueAges==ages[i]).all(axis=1))[0][0]
        protected_attr[i,1] = indx_age
    return data, labels, protected_attr

#%% Home Mortgage Disclosure Act (HMDA) data
def load_hmda_data(path):
    data = pd.read_csv(path)
    labels = data['ACTION']
    data = data.drop(['ACTION'], axis=1)
    
    protected_attribs = ['APP_ETH', 'APP_RACE1','APP_SEX']
    S = (data.loc[:, protected_attribs])
    return data, labels, S.values

#%% demographic parity specific group pre-processing
# need to simplify later
def dp_group(A_train, A_dev, A_test):
    # race
    dp_race_train = np.int64(np.zeros((len(A_train))))
    dp_race_dev = np.int64(np.zeros((len(A_dev))))
    dp_race_test = np.int64(np.zeros((len(A_test))))
    
    # gender
    dp_gender_train = np.int64(np.zeros((len(A_train))))
    dp_gender_dev = np.int64(np.zeros((len(A_dev))))
    dp_gender_test = np.int64(np.zeros((len(A_test))))
    
    # nationality
    dp_nation_train = np.int64(np.zeros((len(A_train))))
    dp_nation_dev = np.int64(np.zeros((len(A_dev))))
    dp_nation_test = np.int64(np.zeros((len(A_test))))
    
    # most underprivileged group
    dp_most_under_train = np.int64(np.zeros((len(A_train))))
    dp_most_under_dev = np.int64(np.zeros((len(A_dev))))
    dp_most_under_test = np.int64(np.zeros((len(A_test))))
    
    # train
    for i in range(len(A_train)):
        if A_train[i,0]==1:
            dp_race_train[i] = 1
        if A_train[i,1]==1:
            dp_gender_train[i] = 1
        if A_train[i,2]==1:
            dp_nation_train[i] = 1
        if A_train[i,0]!=0 or A_train[i,1]!=0 or A_train[i,2]!=0:
            dp_most_under_train[i] = 1
    # dev
    for i in range(len(A_dev)):
        if A_dev[i,0]==1:
            dp_race_dev[i] = 1
        if A_dev[i,1]==1:
            dp_gender_dev[i] = 1
        if A_dev[i,2]==1:
            dp_nation_dev[i] = 1
        if A_dev[i,0]!=0 or A_dev[i,1]!=0 or A_dev[i,2]!=0:
            dp_most_under_dev[i] = 1
    # test
    for i in range(len(A_test)):
        if A_test[i,0]==1:
            dp_race_test[i] = 1
        if A_test[i,1]==1:
            dp_gender_test[i] = 1
        if A_test[i,2]==1:
            dp_nation_test[i] = 1
        if A_test[i,0]!=0 or A_test[i,1]!=0 or A_test[i,2]!=0:
            dp_most_under_test[i] = 1
    return dp_race_train, dp_gender_train, dp_nation_train, dp_most_under_train,dp_race_dev,dp_gender_dev,dp_nation_dev,dp_most_under_dev,dp_race_test,dp_gender_test,dp_nation_test,dp_most_under_test

def dp_group_compas(A_train, A_dev, A_test):
    # race
    dp_race_train = np.int64(np.zeros((len(A_train))))
    dp_race_dev = np.int64(np.zeros((len(A_dev))))
    dp_race_test = np.int64(np.zeros((len(A_test))))
    
    # gender
    dp_gender_train = np.int64(np.zeros((len(A_train))))
    dp_gender_dev = np.int64(np.zeros((len(A_dev))))
    dp_gender_test = np.int64(np.zeros((len(A_test))))
    
    # most underprivileged group
    dp_most_under_train = np.int64(np.zeros((len(A_train))))
    dp_most_under_dev = np.int64(np.zeros((len(A_dev))))
    dp_most_under_test = np.int64(np.zeros((len(A_test))))
    
    # train
    for i in range(len(A_train)):
        if A_train[i,0]==1:
            dp_race_train[i] = 1
        if A_train[i,1]==1:
            dp_gender_train[i] = 1
        if A_train[i,0]!=0 or A_train[i,1]!=0:
            dp_most_under_train[i] = 1
    # dev
    for i in range(len(A_dev)):
        if A_dev[i,0]==1:
            dp_race_dev[i] = 1
        if A_dev[i,1]==1:
            dp_gender_dev[i] = 1
        if A_dev[i,0]!=0 or A_dev[i,1]!=0:
            dp_most_under_dev[i] = 1
    # test
    for i in range(len(A_test)):
        if A_test[i,0]==1:
            dp_race_test[i] = 1
        if A_test[i,1]==1:
            dp_gender_test[i] = 1
        if A_test[i,0]!=0 or A_test[i,1]!=0:
            dp_most_under_test[i] = 1
    return dp_race_train, dp_gender_train, dp_most_under_train,dp_race_dev,dp_gender_dev,dp_most_under_dev,dp_race_test,dp_gender_test,dp_most_under_test
