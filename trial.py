#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 23 13:56:17 2018

@author: tauro
"""
import numpy as np
import pandas as pd

from sklearn.preprocessing import LabelEncoder

import warnings
warnings.filterwarnings('ignore')

import matplotlib.pyplot as plt
import seaborn as sns

train = pd.read_csv("data_train.csv")
test = pd.read_csv("data_test.csv")

train['target'].value_counts()
#There is class imbalance


#dealing with missing values

def missing_values(df):
    #Find total missing values
    mis_val = df.isnull().sum()
    mis_val_percent = 100 * mis_val / len(df)    
    mis_val_table = pd.concat([mis_val, mis_val_percent], axis = 1)    
    mis_val_table = mis_val_table[mis_val_table.iloc[:,1] != 0].sort_values(
            1, ascending = False).round(4)    
    mis_val_table = mis_val_table.rename(columns = {0: 'Total missing', 1: '% missing'})
    
    return mis_val_table

missing = missing_values(train)

train.dtypes

def cat_cols_to_object(df):
    cat_cols = [element for element in list(df) if 'cat' in element]
    
    for col in cat_cols:
        df[col] = df[col].astype(object)
        
    return df


train = cat_cols_to_object(train)
test = cat_cols_to_object(test)

#Number of unique classes in each object predictor
train.select_dtypes('object').apply(pd.Series.nunique, axis = 0).sum()
test.select_dtypes('object').apply(pd.Series.nunique, axis = 0)

missing_cats_train = missing_values(train.filter(regex='cat'))
missing_cats_test = missing_values(test.filter(regex='cat'))

#train_new = pd.DataFrame(index=train['id'])
train_new = pd.DataFrame()

#Encoding categorical variables
le = LabelEncoder()
le_count = 0
ohe_count = 0

for col in train:
    if train[col].dtype == 'object':
        if len(train[col].unique()) <= 2:
           le.fit(train[col])
           le_train = le.transform(train[col])
           #train_new.append({col: le_train}, ignore_index=True)
           del train[col]
           train_new = pd.concat([train_new, pd.DataFrame(le_train)], axis = 1)
           #test[col] = le.transform(test[col])
            
           print(le_train)
           le_count += 1
           
        else:
            temp_col = pd.get_dummies(train[col], drop_first=True)   
            del train[col]
            train_new = pd.concat([train_new, temp_col], axis = 1)
            ohe_count += 1
            
print('%d columns were label encoded' %le_count)

train = pd.concat([train, train_new], axis = 1)

#One-hot encoding
#train = pd.get_dummies(train)
#test = pd.get_dummies(test)

from sklearn.preprocessing import Imputer
imputer = Imputer(strategy = 'median')

train['num18'] = imputer.fit_transform(np.array(train['num18']).reshape(-1,1))

train['num22'] = imputer.fit_transform(np.array(train['num22']).reshape(-1,1))

train['num19'] = imputer.fit_transform(np.array(train['num19']).reshape(-1,1))

train['num20'] = imputer.fit_transform(np.array(train['num20']).reshape(-1,1))




