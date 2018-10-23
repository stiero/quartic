#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  2 14:56:18 2018

@author: tauro
"""


import numpy as np
import pandas as pd


import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import StandardScaler, PowerTransformer, MinMaxScaler, LabelEncoder
from sklearn.metrics import accuracy_score, roc_auc_score, cohen_kappa_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA 

#Progress bar
from tqdm import tqdm
tqdm.pandas(desc="progress-bar")

#Garbage collection
import gc

import os
owd = os.getcwd()





def process_data(train_file, test_file, imbalance_corr = True, **kwargs):
    
    gc.collect()
    
    os.chdir(owd+"/data")
    
    print("\nCommencing data loading and preprocessing\n")
        
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
    
    print("The training data has %d features with missing observations" % len(missing_train))
    print("The test data has %d features with missing observations" % len(missing_test))

    

    train.dtypes
    
    def cat_cols_to_object(df):
        cat_cols = [element for element in list(df) if 'cat' in element]
        
        for col in cat_cols:
            df[col] = df[col].astype(object)
            
        return df
    
    
    train = cat_cols_to_object(train)
    test = cat_cols_to_object(test)
    
    #Number of unique classes in each feature of 'object' type
    train_unique_classes = train.select_dtypes('object').apply(pd.Series.nunique, axis = 0)
    test_unique_classes = test.select_dtypes('object').apply(pd.Series.nunique, axis = 0)
    
    missing_cats_train = missing_values(train.filter(regex='cat'))
    missing_cats_test = missing_values(test.filter(regex='cat'))
    
    #train_new = pd.DataFrame(index=train['id'])
    train_cats_encoded = pd.DataFrame()
    test_cats_encoded = pd.DataFrame()
    
    #Encoding categorical variables
    le = LabelEncoder()
    sc = StandardScaler()
    pt = PowerTransformer()
    
    print("\nEncoding features\n")
    
    mm = MinMaxScaler(feature_range=(-10,10))
    le_count = 0
    ohe_count = 0
    scale_count = 0
    
    
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
            
            if "scale" in kwargs:
                if kwargs.get("scale") == True:
                
                
                    #To suppress type conversion warnings
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        
                        sc.fit(np.array(train[col]).reshape(-1,1))
                    
                        train[col] = sc.transform(np.array(train[col]).reshape(-1,1))
                        test[col] = sc.transform(np.array(test[col]).reshape(-1,1))
                        
                        scale_count += 1
                        
            else:
                pass
                    
            
    print('\n%d categorical features were label encoded' %le_count)
    print('%d categorical features were one-hot encoded' %ohe_count)
    print('%d numerical features were scaled \n' %scale_count)
    
    train = pd.concat([train, train_cats_encoded], axis = 1)
    test = pd.concat([test, test_cats_encoded], axis = 1)
        
    
    def imputer(train):
        
        imputer = SimpleImputer(strategy = 'median')
        
        print("\nImputing missing values in numerical features using their median\n")
        
        cols_train = [str(col) for col in train.columns]
        
        imp_count = 0
        
        
        for col in tqdm(cols_train):
            if "num" in col:
                train[col] = imputer.fit_transform(np.array(train[col]).reshape(-1,1))
                test[col] = imputer.transform(np.array(test[col]).reshape(-1,1))
                imp_count += 1
                
        print('\nMissing values in %d numerical features were imputed\n' %imp_count)
        
        
        return train, test
        
    train, test = imputer(train)


    def compute_pca(df, n_comps):
        
        pca = PCA(n_components = n_comps, copy=True)        
        df = pca.fit_transform(df)              
        percent_var_explained = sum(pca.explained_variance_ratio_)*100
        
        print('\n%d PCA components explain %d%% of the total variance in the original features\n' 
              %(n_comps, percent_var_explained))
        
        return pd.DataFrame(df)

    if "pca" in kwargs:
        if kwargs.get("pca") == True:
            train = compute_pca(train, 100)
    
        
    # Addressing class imbalance
    if imbalance_corr == True:
    
        train['target'] = response
        
        train_maj = train[train['target'] == 0]
        train_min = train[train['target'] == 1]
        
        train_min_sampled = resample(train_min, replace = True,
                                     n_samples = len(train_maj))
        
        train = pd.concat([train_maj, train_min_sampled], ignore_index=True)
        
        response = train['target']
        del train['target']
        
        print("\nAfter adjusting for class imbalance, the the train data dimension is %s\n"
              % (train.shape,))
            
    os.chdir(owd)
    
    print("\nData loading and preprocessing completed\n")
        
    return train, response, test, test_ids
    
    
    


# Train test split for training purposes
def data_split(train, response):
    
    gc.collect()
    
    #os.chdir("./data")
         
    X_train, X_test, y_train, y_test = train_test_split(train, response,
                                                        test_size=0.1)
    
    
    del train
    
    os.chdir(owd)
    
    return X_train, X_test, y_train, y_test

