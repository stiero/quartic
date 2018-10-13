#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  5 00:11:14 2018

@author: tauro
"""

import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier


def timer(start_time=None):
    if not start_time:
        start_time = datetime.now()
        return start_time
    elif start_time:
        thour, temp_sec = divmod((datetime.now() - start_time).total_seconds(), 3600)
        tmin, tsec = divmod(temp_sec, 60)
        print('\n Time taken: %i hours %i minutes and %s seconds.' % (thour, tmin, round(tsec, 2)))


params_xgb = {
        'min_child_weight': [1, 5, 10],
        'gamma': [0.5, 1, 1.5, 2, 5, 10],
        'subsample': [0.6, 0.8, 1.0],
        'colsample_bytree': [0.6, 0.8, 1.0],
        'max_depth': [3, 4, 5],
        'learning_rate': [0.001, 0.01, 0.1]
        }

params_rf = {
        'max_depth':[50, 100, 1000, 10000],
        'min_samples_leaf': [1, 2, 3, 4, 5, 7, 8, 9, 10],
        'max_features': ['auto', 'sqrt', 'log2', None]
        }

rf = RandomForestClassifier(n_estimators = 500, random_state = 50, verbose = 3,
                                       n_jobs = -1, oob_score=True)

lgb = LGBMClassifier(n_estimators=500, objective='binary', class_weight='balanced',
                     learning_rate=0.005, reg_alpha=0.5, reg_lambda=0.3, subsample=0.8,
                     n_jobs=-1, random_state=50)


xgb = XGBClassifier(learning_rate=0.02, n_estimators=600, objective='binary:logistic',
                    silent=True, nthread=-1)

folds = 10
param_comb = 10

skf = StratifiedKFold(n_splits=folds, shuffle = True, random_state = 1001)

random_search = RandomizedSearchCV(rf, param_distributions=params_rf, n_iter=param_comb, 
                                   scoring='roc_auc', n_jobs=4, cv=skf.split(train, response), 
                                   verbose=3, random_state=1001 )

start_time = timer(None)

random_search.fit(train, response)
timer(start_time)


print('\n All results:')
print(random_search.cv_results_)
print('\n Best estimator:')
print(random_search.best_estimator_)
print('\n Best normalized gini score for %d-fold search with %d parameter combinations:' % (folds, param_comb))
print(random_search.best_score_ * 2 - 1)
print('\n Best hyperparameters:')
print(random_search.best_params_)
results = pd.DataFrame(random_search.cv_results_)
results.to_csv('xgb-random-grid-search-results-01.csv', index=False)

y_test = random_search.predict(test)
test['id'] =  test_ids
#results_df = pd.DataFrame(data={'id':test['id'], 'target':y_test[:,1]})
results_df = pd.DataFrame(data={'id':test['id'], 'target':y_test})
results_df.to_csv('submission-random-grid-search-xgb-porto-01.csv', index=False)