#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 10 09:21:59 2018

@author: tauro
"""

from sklearn.ensemble import RandomForestClassifier

list_rf = []

metrics_rf = {}

rf = RandomForestClassifier(n_estimators = 5000, random_state = 50, verbose = 1,
                                       n_jobs = -1, oob_score=True)

#del train['target']

rf.fit(X_train, y_train)

feature_importance_values = rf.feature_importances_

rf_pred = rf.predict(X_test)
    
accuracy = accuracy_score(y_test, rf_pred)
metrics_rf['accuracy'] = accuracy
    
roc = roc_auc_score(y_test, rf_pred)
metrics_rf['auc_roc'] = roc
    
kappa = cohen_kappa_score(y_test, rf_pred)
metrics_rf['kappa'] = kappa

conf_matrix = confusion_matrix(y_test, rf_pred)
metrics_rf['conf_matrix'] = conf_matrix

metrics_rf['oob_score'] = rf.oob_score_

##########################################################################################

metrics_rf = {}

rf = RandomForestClassifier(n_estimators = 5000, random_state = 50, verbose = 1, n_jobs = -1, 
                            oob_score=True, min_samples_split = 10, min_samples_leaf = 5)

#del train['target']

rf.fit(X_train, y_train)

feature_importance_values = rf.feature_importances_

rf_pred = rf.predict(X_test)
    
accuracy = accuracy_score(y_test, rf_pred)
metrics_rf['accuracy'] = accuracy
    
roc = roc_auc_score(y_test, rf_pred)
metrics_rf['auc_roc'] = roc
    
kappa = cohen_kappa_score(y_test, rf_pred)
metrics_rf['kappa'] = kappa

conf_matrix = confusion_matrix(y_test, rf_pred)
metrics_rf['conf_matrix'] = conf_matrix

metrics_rf['oob_score'] = rf.oob_score_

##########################################################################################

metrics_rf = {}

rf = RandomForestClassifier(n_estimators = 5000, random_state = 50, verbose = 1,
                                       n_jobs = -1, oob_score=True, max_features = "sqrt")

#del train['target']

rf.fit(X_train, y_train)

feature_importance_values = rf.feature_importances_

rf_pred = rf.predict(X_test)
    
accuracy = accuracy_score(y_test, rf_pred)
metrics_rf['accuracy'] = accuracy
    
roc = roc_auc_score(y_test, rf_pred)
metrics_rf['auc_roc'] = roc
    
kappa = cohen_kappa_score(y_test, rf_pred)
metrics_rf['kappa'] = kappa

conf_matrix = confusion_matrix(y_test, rf_pred)
metrics_rf['conf_matrix'] = conf_matrix

metrics_rf['oob_score'] = rf.oob_score_

##########################################################################################

metrics_rf = {}

rf = RandomForestClassifier(n_estimators = 5000, random_state = 50, verbose = 1,
                                       n_jobs = -1, oob_score=True, min_samples_split = 10,
                                       min_samples_leaf = 5, max_features = "sqrt")

#del train['target']

rf.fit(X_train, y_train)

feature_importance_values = rf.feature_importances_

rf_pred = rf.predict(X_test)
    
accuracy = accuracy_score(y_test, rf_pred)
metrics_rf['accuracy'] = accuracy
    
roc = roc_auc_score(y_test, rf_pred)
metrics_rf['auc_roc'] = roc
    
kappa = cohen_kappa_score(y_test, rf_pred)
metrics_rf['kappa'] = kappa

conf_matrix = confusion_matrix(y_test, rf_pred)
metrics_rf['conf_matrix'] = conf_matrix

metrics_rf['oob_score'] = rf.oob_score_

##########################################################################################

metrics_rf = {}

rf = RandomForestClassifier(n_estimators = 5000, random_state = 50, verbose = 1,
                                       n_jobs = -1, oob_score=True, max_features = "log2")

#del train['target']

rf.fit(X_train, y_train)

feature_importance_values = rf.feature_importances_

rf_pred = rf.predict(X_test)
    
accuracy = accuracy_score(y_test, rf_pred)
metrics_rf['accuracy'] = accuracy
    
roc = roc_auc_score(y_test, rf_pred)
metrics_rf['auc_roc'] = roc
    
kappa = cohen_kappa_score(y_test, rf_pred)
metrics_rf['kappa'] = kappa

conf_matrix = confusion_matrix(y_test, rf_pred)
metrics_rf['conf_matrix'] = conf_matrix

metrics_rf['oob_score'] = rf.oob_score_

##########################################################################################
metrics_rf = {}

rf = RandomForestClassifier(n_estimators = 5000, random_state = 50, verbose = 1,
                                       n_jobs = -1, oob_score=True, min_samples_split = 10,
                                       min_samples_leaf = 5, max_features = "log2")

#del train['target']

rf.fit(X_train, y_train)

feature_importance_values = rf.feature_importances_

rf_pred = rf.predict(X_test)
    
accuracy = accuracy_score(y_test, rf_pred)
metrics_rf['accuracy'] = accuracy
    
roc = roc_auc_score(y_test, rf_pred)
metrics_rf['auc_roc'] = roc
    
kappa = cohen_kappa_score(y_test, rf_pred)
metrics_rf['kappa'] = kappa

conf_matrix = confusion_matrix(y_test, rf_pred)
metrics_rf['conf_matrix'] = conf_matrix

metrics_rf['oob_score'] = rf.oob_score_

##########################################################################################