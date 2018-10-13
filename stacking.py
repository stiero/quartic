#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 12 14:48:43 2018

@author: ishippoml
"""

from sklearn.ensemble import VotingClassifier
from scipy.stats import mode

from sklearn.naive_bayes import GaussianNB

list_gnb = []

metrics_gnb = {}

gnb = GaussianNB()

gnb.fit(X_train, y_train)

gnb_pred = pd.Series(gnb.predict(X_test))

accuracy = accuracy_score(y_test, gnb_pred)
metrics_gnb['accuracy'] = accuracy
    
roc = roc_auc_score(y_test, gnb_pred)
metrics_gnb['auc_roc'] = roc
    
kappa = cohen_kappa_score(y_test, gnb_pred)
metrics_gnb['kappa'] = kappa

conf_matrix = confusion_matrix(y_test, gnb_pred)
metrics_gnb['conf_matrix'] = conf_matrix

list_gnb.append(metrics_gnb)

#X_train['gnb_pred'] = gnb_pred

########################################################

from sklearn.ensemble import RandomForestClassifier

list_rf = []

metrics_rf = {}

rf = RandomForestClassifier(n_estimators = 5000, random_state = 50, verbose = 1,
                                       n_jobs = -1, oob_score=True)

#del train['target']

rf.fit(X_train, y_train)

feature_importance_values = rf.feature_importances_

rf_pred = rf.predict(X_test)
rf_pred = pd.Series(rf_pred)

accuracy = accuracy_score(y_test, rf_pred)
metrics_rf['accuracy'] = accuracy
    
roc = roc_auc_score(y_test, rf_pred)
metrics_rf['auc_roc'] = roc
    
kappa = cohen_kappa_score(y_test, rf_pred)
metrics_rf['kappa'] = kappa

conf_matrix = confusion_matrix(y_test, rf_pred)
metrics_rf['conf_matrix'] = conf_matrix

metrics_rf['oob_score'] = rf.oob_score_

list_rf.append(metrics_rf)

#########################################################

import xgboost as xgb

list_xgb = []


dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)

params = {'max_depth': 2, 'eta': 0.5, 'silent': 0, 'objective': 'binary:logistic',
          'nthread': 4, 'eval_metric': 'auc', 'colsample_bytree': 0.8, 'subsample': 0.8, 
          'scale_pos_weight': 1, 'gamma': 200, 'learning_rate': 0.02}


evallist = [(dtest, 'eval'), (dtrain, 'train')]

num_rounds = 5000


bst = xgb.train(params, dtrain, num_rounds, evallist)

bst_pred = bst.predict(dtest)

threshold = 0.5

xgb_pred = bst_pred > threshold
xgb_pred = np.multiply(xgb_pred, 1)

xgb_pred = pd.Series(xgb_pred)

metrics_xgb = {}

metrics_xgb['num_rounds'] = num_rounds
metrics_xgb['params'] = params

accuracy = accuracy_score(y_test, xgb_pred)
metrics_xgb['accuracy'] = accuracy
    
roc = roc_auc_score(y_test, xgb_pred)
metrics_xgb['auc_roc'] = roc
    
kappa = cohen_kappa_score(y_test, xgb_pred)
metrics_xgb['kappa'] = kappa

conf_matrix = confusion_matrix(y_test, xgb_pred)
metrics_xgb['conf_matrix'] = conf_matrix

metrics_xgb['threshold'] = threshold

metrics_xgb['sensitivity'] = conf_matrix[1,1] / (conf_matrix[1,1] + conf_matrix[1,0])
metrics_xgb['specificity'] = conf_matrix[0,0] / (conf_matrix[0,1] + conf_matrix[0,0])

list_xgb.append(metrics_xgb)

#X_train['xgb_pred']


#################################################

from lightgbm import LGBMClassifier

list_lgb = []

metrics_lgb = {}

lgb = LGBMClassifier(n_estimators=5000, objective='binary', class_weight='balanced',
                     learning_rate=0.005, reg_alpha=0.5, reg_lambda=0.3, subsample=0.8,
                     n_jobs=-1, random_state=50)

lgb.fit(X_train, y_train, eval_metric='auc', verbose=True)

lgb_pred = lgb.predict(X_test)
#lgb_pred_prob = lgb.predict_proba(X_test)[:,1]

threshold = 0.5

#lgb_pred = lgb_pred_prob > threshold

accuracy = accuracy_score(y_test, lgb_pred)
metrics_lgb['accuracy'] = accuracy
    
roc = roc_auc_score(y_test, lgb_pred)
metrics_lgb['auc_roc'] = roc
    
kappa = cohen_kappa_score(y_test, lgb_pred)
metrics_lgb['kappa'] = kappa

conf_matrix = confusion_matrix(y_test, lgb_pred)
metrics_lgb['conf_matrix'] = conf_matrix

list_lgb.append(metrics_lgb)

###################################################

from sklearn.svm import SVC

list_svm = []

svc = SVC(C=1, kernel='rbf', degree=3)

svc.fit(X_train, y_train)

svc_pred = svc.predict(X_test)

metrics_svm = {}

metrics_svm['accuracy'] = accuracy(y_test, svc_pred)
metrics_svm['auc_roc'] = roc_auc_score(y_test, svc_pred)
metrics_svm['kappa'] = cohen_kappa_score(y_test, svc_pred)
metrics_svm['conf_matrix'] = confusion_matrix(y_test, svc_pred)

list_svm.append(metrics_svm)

########################################################

#ens = VotingClassifier(estimators=[('gnb', gnb), ('rf', rf), ('xgb', xgb)])

#ens.fit(X_train, y_train)

#ens.score(X_test, y_test)

final_pred = np.array([])
for i in range(len(X_test)):
    final_pred = np.append(final_pred, mode([gnb_pred[i], rf_pred[i], 
                                             xgb_pred[i], lgb_pred[i],
                                             svc_pred[i]])[0].item())
    

list_final = []

metrics_final = {}

metrics_final['num_rounds'] = num_rounds
metrics_final['params'] = params

accuracy = accuracy_score(y_test, final_pred)
metrics_final['accuracy'] = accuracy
    
roc = roc_auc_score(y_test, final_pred)
metrics_final['auc_roc'] = roc
    
kappa = cohen_kappa_score(y_test, final_pred)
metrics_final['kappa'] = kappa

conf_matrix = confusion_matrix(y_test, final_pred)
metrics_final['conf_matrix'] = conf_matrix

metrics_final['threshold'] = threshold

metrics_final['sensitivity'] = conf_matrix[1,1] / (conf_matrix[1,1] + conf_matrix[1,0])
metrics_final['specificity'] = conf_matrix[0,0] / (conf_matrix[0,1] + conf_matrix[0,0])

list_final.append(metrics_final)