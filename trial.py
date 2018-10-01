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

from sklearn.preprocessing import StandardScaler

train = pd.read_csv("data_train.csv")
test = pd.read_csv("data_test.csv")

response = train['target']
del train['target']
del train['id']
del test['id']

response.value_counts()
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

missing_train = missing_values(train)
missing_test = missing_values(test)

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
train_cats_encoded = pd.DataFrame()
test_cats_encoded = pd.DataFrame()

#Encoding categorical variables
le = LabelEncoder()
sc = StandardScaler()
le_count = 0
ohe_count = 0

for col in train.columns:
    if train[col].dtype == 'object':
        if len(train[col].unique()) <= 2:
           le.fit(train[col])
           le_train = le.transform(train[col])
           le_test = le.transform(test[col])
           #train_new.append({col: le_train}, ignore_index=True)
           del train[col]
           del test[col]
           
           train_cats_encoded = pd.concat([train_cats_encoded, 
                                           pd.DataFrame(le_train)], axis = 1)
           test_cats_encoded = pd.concat([test_cats_encoded, 
                                          pd.DataFrame(le_test)], axis = 1)
    
           le_count += 1
           
        else:
            ohe_train = pd.get_dummies(train[col], drop_first=True) 
            ohe_test = pd.get_dummies(test[col], drop_first=True)
            
            del train[col]
            del test[col]
            
            train_cats_encoded = pd.concat([train_cats_encoded, ohe_train], axis = 1)
            test_cats_encoded = pd.concat([test_cats_encoded, ohe_test], axis = 1)
            
            ohe_count += 1
            
    else:
        sc.fit(np.array(train[col]).reshape(-1,1))
        
        train[col] = sc.transform(np.array(train[col]).reshape(-1,1))
        test[col] = sc.transform(np.array(test[col]).reshape(-1,1))
            
        
print('%d categorical columns were label encoded' %le_count)
print('%d categorical columns were one-hot encoded' %ohe_count)

train = pd.concat([train, train_cats_encoded], axis = 1)
test = pd.concat([test, test_cats_encoded], axis = 1)


#Umputing missing int/float values
from sklearn.preprocessing import Imputer
imputer = Imputer(strategy = 'median')

#train['num18'] = imputer.fit_transform(np.array(train['num18']).reshape(-1,1))
#test['num18'] = imputer.transform(np.array(test['num18']).reshape(-1,1))

#train['num22'] = imputer.fit_transform(np.array(train['num22']).reshape(-1,1))
#test['num22'] = imputer.transform(np.array(test['num22']).reshape(-1,1))

#train['num19'] = imputer.fit_transform(np.array(train['num19']).reshape(-1,1))
#test['num19'] = imputer.transform(np.array(test['num19']).reshape(-1,1))

#train['num20'] = imputer.fit_transform(np.array(train['num20']).reshape(-1,1))
#test['num20'] = imputer.transform(np.array(test['num20']).reshape(-1,1))

cols_train = [str(col) for col in train.columns]

for col in cols_train:
    if "num" in col:
        train[col] = imputer.fit_transform(np.array(train[col]).reshape(-1,1))
        test[col] = imputer.transform(np.array(test[col]).reshape(-1,1))

train['target'] = response

###################################################

#Logistic regression

from sklearn.linear_model import LogisticRegression

log_reg = LogisticRegression(C=0.001)

log_reg.fit(train, response)

log_reg_pred = log_reg.predict(test)



#####################################################

# 




