#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  5 14:56:18 2018

@author: ishippoml
"""


import numpy as np
import pandas as pd

from sklearn.preprocessing import LabelEncoder

import warnings
warnings.filterwarnings('ignore')

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler

from sklearn.metrics import accuracy_score, roc_auc_score, cohen_kappa_score, confusion_matrix

from sklearn.model_selection import train_test_split

from sklearn.utils import resample

from tqdm import tqdm
tqdm.pandas(desc="progress-bar")

from sklearn.preprocessing import Imputer

from sklearn.decomposition import PCA    



def process_data(train_file, test_file, pca):
    
    #Only works on an IPython interpreter
    %reset -f array
        
    train = pd.read_csv(train_file)
    test = pd.read_csv(test_file)
    
    
    response = train['target']
    train_ids = train['id']
    test_ids = test['id']
    
    del train['target']
    del train['id']
    del test['id']
    
    train_cols = list(train)
    test_cols = list(test)
    
    response.value_counts()
    
    
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
    
    for col in tqdm(train.columns):
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
                ohe_train = pd.get_dummies(train[col], drop_first=True, prefix=col) 
                ohe_test = pd.get_dummies(test[col], drop_first=True, prefix=col)
                
                del train[col]
                del test[col]
                
                train_cats_encoded = pd.concat([train_cats_encoded, ohe_train], axis = 1)
                test_cats_encoded = pd.concat([test_cats_encoded, ohe_test], axis = 1)
                
                ohe_count += 1
                
        else:
            
            #To suppress type conversion warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                
                sc.fit(np.array(train[col]).reshape(-1,1))
            
                train[col] = sc.transform(np.array(train[col]).reshape(-1,1))
                test[col] = sc.transform(np.array(test[col]).reshape(-1,1))
                
            
    print('%d categorical columns were label encoded' %le_count)
    print('%d categorical columns were one-hot encoded' %ohe_count)
    
    train = pd.concat([train, train_cats_encoded], axis = 1)
    test = pd.concat([test, test_cats_encoded], axis = 1)
        
    
    def imputer(train):
        imputer = Imputer(strategy = 'median')
        
        cols_train = [str(col) for col in train.columns]
        
        imp_count = 0
        
        for col in tqdm(cols_train):
            if "num" in col:
                train[col] = imputer.fit_transform(np.array(train[col]).reshape(-1,1))
                test[col] = imputer.transform(np.array(test[col]).reshape(-1,1))
                imp_count += 1
                
        print('Missing values in %d columns were imputed' %imp_count)
        
        
        return train, test
        
    train, test = imputer(train)


    def compute_pca(df, n_comps):
        
        pca = PCA(n_components = n_comps, copy=True)        
        
        df = train.copy()        
        del df['target']        
        
        df = pca.fit_transform(df)        
        df = np.column_stack((df, np.array(response)))
        
        print(pca.explained_variance_ratio_)        
        print(sum(pca.explained_variance_ratio_))
        
        return pd.DataFrame(df)

    if pca == True:
        train = compute_pca(train, 50)
        
    return train, response
    
    
    


# Train test split   
def data_split(train, response, imbalance_corr):

#    try:
#        del train['target']
#    except KeyError:
#        train['target'] = response
#            
    X_train, X_test, y_train, y_test = train_test_split(train, response,
                                                        test_size=0.1, random_state = 3536)
    
    
    # Addressing class imbalance
    
    if imbalance_corr == True:
    
        X_train['target'] = y_train
        
        train_maj = X_train[X_train['target'] == 0]
        train_min = X_train[X_train['target'] == 1]
        
        train_min_sampled = resample(train_min, replace = True,
                                     n_samples = len(train_maj), random_state = 9868)
        
        X_train = pd.concat([train_maj, train_min_sampled])
        
        y_train = X_train['target']
        del X_train['target']
    
    del train
    
    return X_train, X_test, y_train, y_test



train, response = process_data("data_train.csv", "data_test.csv", pca=False)
X_train, X_test, y_train, y_test = data_split(train, response, imbalance_corr=False)
