#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 10 20:39:45 2018

@author: tauro
"""

from sklearn.feature_selection import SelectKBest, chi2, f_classif

kbest = SelectKBest(score_func=f_classif, k=100)

#X_train = np.array(X_train.values)

kbest.fit(train, response)

kbest_train = kbest.transform(train)
#kbest_test = kbest.transform(test)

#del train, test, test_ids, response

####################################################

from xgboost import XGBClassifier

list_xgb = []

xgb_new = XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
       colsample_bytree=1.0, gamma=5, learning_rate=0.1, max_delta_step=0,
       max_depth=5, min_child_weight=10, missing=None, n_estimators=600,
       n_jobs=4, objective='binary:logistic', random_state=1001,
       reg_alpha=0, reg_lambda=1, scale_pos_weight=1, seed=None,
       silent=True, subsample=1.0)

xgb_new.fit(kbest_train, y_train)

xgb_new_pred = xgb_new.predict(kbest_test)

metrics_xgb = {}

metrics_xgb['accuracy'] = accuracy_score(y_test, xgb_new_pred)
metrics_xgb['auc_roc'] = roc_auc_score(y_test, xgb_new_pred)
metrics_xgb['kappa'] = cohen_kappa_score(y_test, xgb_new_pred)
metrics_xgb['conf_matrix'] = confusion_matrix(y_test, xgb_new_pred)

list_xgb.append(metrics_xgb)
