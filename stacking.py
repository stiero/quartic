#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 12 14:48:43 2018

@author: tauro
"""

import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler, PowerTransformer, MinMaxScaler, LabelEncoder
from sklearn.metrics import accuracy_score, roc_auc_score, cohen_kappa_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold    
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier


from lightgbm import LGBMClassifier

from xgboost import XGBClassifier

from scipy.stats import mode

from keras.models import Sequential
from keras.layers import Dense, Dropout


import gc
import os
from datetime import datetime

from tqdm import tqdm
tqdm.pandas(desc="progress-bar")

#Custom file for data loading and preprocessing
from data_preprocessing import process_data

start_time = datetime.now()

owd = os.getcwd()





train, response, test, test_ids = process_data("data_train.csv", "data_test.csv", 
                                               pca=False, scale=True)






def stacking(model, train, response, test, test_ids, n_fold):

    kfold = StratifiedKFold(n_splits=n_fold, shuffle=True, random_state=1986)
    
    col_name = str(model)[:10]
    
    list_model = []
        
    train_new = pd.DataFrame(columns=[col_name])
    
    test_new = []
    
    
    
    for tr, te in tqdm(kfold.split(train, response)):
        gc.collect()
        
        train_cv = train.iloc[tr]
        test_cv = train.iloc[te]
        
        response_train = response.iloc[tr]
        response_test = response.iloc[te]
        

        
        metrics_model = {}
        
    
        model.fit(train_cv, response_train)
        model_pred = pd.DataFrame({col_name: model.predict(test_cv)}, index=te)
        model_pred[col_name] = model_pred[col_name].astype(int)        
       

        model_pred_test = model.predict(test)

        
        accuracy = accuracy_score(response_test, model_pred)
        metrics_model['accuracy'] = accuracy
            
        roc = roc_auc_score(response_test, model_pred)
        metrics_model['auc_roc'] = roc
            
        kappa = cohen_kappa_score(response_test, model_pred)
        metrics_model['kappa'] = kappa
        
        conf_matrix = confusion_matrix(response_test, model_pred)
        metrics_model['conf_matrix'] = conf_matrix
        
        list_model.append(metrics_model)
        
        train_new = pd.concat([train_new, model_pred], axis=0)
        
        test_new.append(model_pred_test)
        
        
    
    test_new = np.array(test_new)
    
    new_test_col = mode(test_new)[0]
    
    new_test_col = pd.DataFrame(new_test_col.T, columns = [col_name])
    new_test_col[col_name] = new_test_col[col_name].astype(int)
    
    
    train = pd.concat([train, train_new], axis = 1)
    test = pd.concat([test, new_test_col], axis = 1)
         
    return train, test, list_model



params = {'max_depth': 2, 'eta': 0.5, 'silent': 0, 'objective': 'binary:logistic',
          'nthread': 4, 'eval_metric': 'auc', 'colsample_bytree': 0.8, 'subsample': 0.8, 
          'scale_pos_weight': 26, 'gamma': 200, 'learning_rate': 0.02}



gnb = GaussianNB()

train, test, list_gnb = stacking(gnb, train, response, test, test_ids, n_fold=10)
del gnb    


rf = RandomForestClassifier(n_estimators = 500, random_state = 50, verbose = 1,
                                       n_jobs = -1, oob_score=True)

train, test, list_rf = stacking(rf, train, response, test, test_ids, n_fold=10)
del rf


lgb = LGBMClassifier(n_estimators=500, objective='binary', class_weight='balanced',
                     learning_rate=0.005, reg_alpha=0.5, reg_lambda=0.3, subsample=0.8,
                     n_jobs=-1, random_state=50)

train, test, list_lgb = stacking(lgb, train, response, test, test_ids, n_fold=10)
del lgb






#gnb = pickle.load(open(owd+"/models/gnb.pkl", 'rb'))
#
#
#rf = joblib.load(owd+"/models/rf.joblib")
#
#
#lgb = pickle.load(open(owd+"/models/lgb.pkl", 'rb'))
#
#
#adb = pickle.load(open(owd+"/models/adb.pkl", 'rb'))
#
#
#bst = pickle.load(open(owd+"/models/xgb.pkl", 'rb'))
#
#
#nn = load_model(owd+"/models/nn.h5")







    

    
