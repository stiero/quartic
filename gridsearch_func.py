#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct  6 14:16:58 2018

@author: ishippoml
"""

import sys
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd

from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

def gridsearch(algo, train, response):
    
    params_xgb = {
        'min_child_weight': [1, 5, 10],
        'gamma': [0.5, 1, 1.5, 2, 5],
        'subsample': [0.6, 0.8, 1.0],
        'colsample_bytree': [0.6, 0.8, 1.0],
        'max_depth': [3, 4, 5]
        }
    
    params_lgb = {
            'reg_lambda': [0.1, 0.2, 0.5, 1],
            'reg_alpha': [0.1, 0.2, 0.5, 1],
            'learning_rate': [0.01, 0.05, 0.1, 0.5],
            'subsample': [0.5, 0.8, 1]}
    
    if algo == "xgb":
        params = params_xgb
        
        mdl = XGBClassifier(learning_rate=0.02, n_estimators=600, objective='binary:logistic',
                        silent=True, nthread=1)
        
    elif algo == 'lgb':
        params = params_lgb
        
        mdl = LGBMClassifier(n_estimators=3000, objective='binary', class_weight=None, 
                             n_jobs=-1, random_state=50, scale_pos_weight= 26)
        
    else:
        print("Model does not exist")
        sys.exit()
        
    folds = 3
    param_comb = 5
    
    skf = StratifiedKFold(n_splits=folds, shuffle = True, random_state = 1001)

    random_search = RandomizedSearchCV(mdl, param_distributions=params, n_iter=param_comb, 
                                       scoring='roc_auc', n_jobs=4, cv=skf.split(train, response), 
                                       verbose=3, random_state=1001)
    
    random_search.fit(train, response)
    
    print(random_search.cv_results_)
    print(random_search.best_estimator_)
    print(random_search.best_score_ * 2 - 1)
    print(random_search.best_params_)
    
    results = pd.DataFrame(random_search.cv_results_)
    
    return random_search, results
    

model, results = gridsearch("lgb", train, response)
    