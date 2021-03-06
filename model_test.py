#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 1 22:13:05 2018

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
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

from datetime import datetime
import os
import pickle

#Progress bar
from tqdm import tqdm
tqdm.pandas(desc="progress-bar")

#Garbage collection
import gc


from scipy.stats import mode
import xgboost as xgb
from lightgbm import LGBMClassifier


from keras.models import Sequential
from keras.layers import Dense, Dropout

#Custom file for data loading and preprocessing
from data_preprocessing import process_data, data_split


start_time = datetime.now()
owd = os.getcwd()

# =============================================================================
# Load required data 


train, response, test, test_ids = process_data("data_train.csv", "data_test.csv", 
                                               pca=False, scale=True)

X_train, X_test, y_train, y_test = data_split(train, response)

del train, response, test, test_ids





# =============================================================================
# Model 1 - Gaussian Naive Bayes classifier

gc.collect()

print("\nTraining Gaussian Naive Bayes classifier - Model 1 of 8\n")

list_gnb = []

gnb = GaussianNB()

gnb.fit(X_train, y_train)

gnb_pred_prob = gnb.predict_proba(X_test)[:,1]

threshold_gnb = 0.3

gnb_pred = gnb_pred_prob > threshold_gnb

metrics_gnb = {}

metrics_gnb['threshold'] = threshold_gnb

accuracy = accuracy_score(y_test, gnb_pred)
metrics_gnb['accuracy'] = accuracy
    
roc = roc_auc_score(y_test, gnb_pred)
metrics_gnb['auc_roc'] = roc
    
kappa = cohen_kappa_score(y_test, gnb_pred)
metrics_gnb['kappa'] = kappa

conf_matrix = confusion_matrix(y_test, gnb_pred)
metrics_gnb['conf_matrix'] = conf_matrix

metrics_gnb['sensitivity'] = conf_matrix[1,1] / (conf_matrix[1,1] + conf_matrix[1,0])
metrics_gnb['specificity'] = conf_matrix[0,0] / (conf_matrix[0,1] + conf_matrix[0,0])

list_gnb.append(metrics_gnb)
 





# =============================================================================
# Model 2 - XGBoost classifier 

gc.collect()

print("\nTraining XG Boost classifier - Model 2 of 8\n")

list_xgb = []


dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)

params = {'max_depth': 2, 'eta': 0.5, 'silent': 0, 'objective': 'binary:logistic',
          'nthread': 4, 'eval_metric': 'auc', 'colsample_bytree': 0.8, 'subsample': 0.8, 
          'scale_pos_weight': 1, 'gamma': 200, 'learning_rate': 0.02}


evallist = [(dtest, 'eval'), (dtrain, 'train')]

num_rounds = 500


bst = xgb.train(params, dtrain, num_rounds, evallist)

xgb_pred_prob = bst.predict(dtest)

threshold_xgb = 0.5


xgb_pred = xgb_pred_prob > threshold_xgb
xgb_pred = np.multiply(xgb_pred, 1)

xgb_pred = pd.Series(xgb_pred)

metrics_xgb = {}

metrics_xgb['num_rounds'] = num_rounds
metrics_xgb['params'] = params
metrics_xgb['threshold'] = threshold_xgb

accuracy = accuracy_score(y_test, xgb_pred)
metrics_xgb['accuracy'] = accuracy
    
roc = roc_auc_score(y_test, xgb_pred)
metrics_xgb['auc_roc'] = roc
    
kappa = cohen_kappa_score(y_test, xgb_pred)
metrics_xgb['kappa'] = kappa

conf_matrix = confusion_matrix(y_test, xgb_pred)
metrics_xgb['conf_matrix'] = conf_matrix

metrics_xgb['sensitivity'] = conf_matrix[1,1] / (conf_matrix[1,1] + conf_matrix[1,0])
metrics_xgb['specificity'] = conf_matrix[0,0] / (conf_matrix[0,1] + conf_matrix[0,0])

list_xgb.append(metrics_xgb)






###############################################################################

# Model 3 - Random Forest classifier

gc.collect()

print("\nTraining XG Boost classifier - Model 3 of 8\n")


list_rf = []

rf = RandomForestClassifier(n_estimators = 500, random_state = 50, verbose = 1,
                                       n_jobs = -1, oob_score=True)

#del train['target']

rf.fit(X_train, y_train)

feature_importance_values = rf.feature_importances_

rf_pred_prob = rf.predict_proba(X_test)[:,1]

threshold_rf = 0.4

rf_pred = rf_pred_prob > threshold_rf

metrics_rf = {}

accuracy = accuracy_score(y_test, rf_pred)
metrics_rf['accuracy'] = accuracy
    
roc = roc_auc_score(y_test, rf_pred)
metrics_rf['auc_roc'] = roc
    
kappa = cohen_kappa_score(y_test, rf_pred)
metrics_rf['kappa'] = kappa

conf_matrix = confusion_matrix(y_test, rf_pred)
metrics_rf['conf_matrix'] = conf_matrix

metrics_rf['oob_score'] = rf.oob_score_

metrics_rf['threshold'] = threshold_rf

metrics_rf['sensitivity'] = conf_matrix[1,1] / (conf_matrix[1,1] + conf_matrix[1,0])
metrics_rf['specificity'] = conf_matrix[0,0] / (conf_matrix[0,1] + conf_matrix[0,0])

metrics_rf['feature_imp'] = feature_importance_values

list_rf.append(metrics_rf)







# =============================================================================
# Model 4 - Light Gradient Boosting Machine Classifier 

gc.collect()

print("\nTraining LGBM classifier - Model 4 of 8\n")

list_lgb = []


lgb = LGBMClassifier(n_estimators=500, objective='binary', class_weight='balanced',
                     learning_rate=0.005, reg_alpha=0.5, reg_lambda=0.3, subsample=0.8,
                     n_jobs=-1, random_state=50)

lgb.fit(X_train, y_train, eval_metric='auc', verbose=True)

#lgb_pred = lgb.predict(X_test)
lgb_pred_prob = lgb.predict_proba(X_test)[:,1]

threshold_lgb = 0.4

lgb_pred = lgb_pred_prob > threshold_lgb

metrics_lgb = {}

metrics_lgb['threshold'] = threshold_lgb

accuracy = accuracy_score(y_test, lgb_pred)
metrics_lgb['accuracy'] = accuracy
    
roc = roc_auc_score(y_test, lgb_pred)
metrics_lgb['auc_roc'] = roc
    
kappa = cohen_kappa_score(y_test, lgb_pred)
metrics_lgb['kappa'] = kappa

conf_matrix = confusion_matrix(y_test, lgb_pred)
metrics_lgb['conf_matrix'] = conf_matrix

metrics_lgb['threshold'] = threshold_lgb

metrics_lgb['sensitivity'] = conf_matrix[1,1] / (conf_matrix[1,1] + conf_matrix[1,0])
metrics_lgb['specificity'] = conf_matrix[0,0] / (conf_matrix[0,1] + conf_matrix[0,0])


list_lgb.append(metrics_lgb)







# =============================================================================
# Model 5 - Adaptive Boosting Classifier

gc.collect()

print("\nTraining AdaBoost classifier - Model 5 of 8\n")

list_adb = []

adb = AdaBoostClassifier(n_estimators = 500, learning_rate = 0.76, algorithm = 'SAMME.R')

adb.fit(X_train, y_train)

adb_pred_prob = adb.predict_proba(X_test)[:,1]

threshold_adb = 0.4

adb_pred = adb_pred_prob > threshold_adb

metrics_adb = {}

metrics_adb['threshold'] = threshold_adb

accuracy = accuracy_score(y_test, adb_pred)
metrics_adb['accuracy'] = accuracy
    
roc = roc_auc_score(y_test, adb_pred)
metrics_adb['auc_roc'] = roc
    
kappa = cohen_kappa_score(y_test, adb_pred)
metrics_adb['kappa'] = kappa

conf_matrix = confusion_matrix(y_test, adb_pred)
metrics_adb['conf_matrix'] = conf_matrix

metrics_adb['threshold'] = threshold_adb

metrics_adb['sensitivity'] = conf_matrix[1,1] / (conf_matrix[1,1] + conf_matrix[1,0])
metrics_adb['specificity'] = conf_matrix[0,0] / (conf_matrix[0,1] + conf_matrix[0,0])


list_adb.append(metrics_adb)







# =============================================================================
# Model 6 - Multilayer Perceptron Neural Network Classifier

gc.collect()

print("\nTraining MLP classifier - Model 6 of 8\n")


list_nn = []

model = Sequential()
model.add(Dense(100, input_dim=203, activation = 'relu'))
model.add(Dropout(0.5))
model.add(Dense(100, activation = 'relu'))
model.add(Dropout(0.5))
model.add(Dense(100, activation = 'relu'))
model.add(Dropout(0.5))
model.add(Dense(100, activation = 'relu'))
model.add(Dropout(0.5))
model.add(Dense(100, activation = 'relu'))
model.add(Dropout(0.5))
#model.add(Dense(100, activation = 'relu'))
#model.add(Dropout(0.5))
model.add(Dense(1, activation = 'sigmoid'))

model.compile(loss='binary_crossentropy', optimizer = 'adam', 
              metrics = ['accuracy'])

model.fit(X_train, y_train, epochs = 10, batch_size = 32)

nn_pred_prob = model.predict(X_test)

threshold_nn = 0.4

nn_pred = nn_pred_prob > threshold_nn
nn_pred = list(map(int, nn_pred))

metrics_nn = {}

metrics_nn['threshold'] = threshold_nn
metrics_nn['accuracy'] = accuracy_score(y_test, nn_pred)
metrics_nn['auc_roc'] = roc_auc_score(y_test, nn_pred)
metrics_nn['kappa'] = cohen_kappa_score(y_test, nn_pred)
conf_matrix = confusion_matrix(y_test, nn_pred)
metrics_nn['conf_matrix'] = conf_matrix
metrics_nn['sensitivity'] = conf_matrix[1,1] / (conf_matrix[1,1] + conf_matrix[1,0])
metrics_nn['specificity'] = conf_matrix[0,0] / (conf_matrix[0,1] + conf_matrix[0,0])

list_nn.append(metrics_nn)





# =============================================================================
# Model 7 - Discriminant analysis

gc.collect()

list_qda = []

print("\nTraining Quadratic discriminant analysis classifier - Model 7 of 8\n")


qda = QuadraticDiscriminantAnalysis(reg_param=0.0001, tol=0.0001)

qda.fit(X_train, y_train)

qda_pred_prob = qda.predict_proba(X_test)[:,1]

threshold_qda = 0.0001

qda_pred = qda_pred_prob > threshold_qda

metrics_qda = {}

metrics_qda['params'] = str(qda.get_params)
metrics_qda['threshold'] = threshold_qda
accuracy = accuracy_score(y_test, qda_pred)
metrics_qda['accuracy'] = accuracy
roc = roc_auc_score(y_test, qda_pred)
metrics_qda['auc_roc'] = roc
kappa = cohen_kappa_score(y_test, qda_pred)
metrics_qda['kappa'] = kappa
conf_matrix = confusion_matrix(y_test, qda_pred)
metrics_qda['conf_matrix'] = conf_matrix
metrics_qda['threshold'] = threshold_qda
metrics_qda['sensitivity'] = conf_matrix[1,1] / (conf_matrix[1,1] + conf_matrix[1,0])
metrics_qda['specificity'] = conf_matrix[0,0] / (conf_matrix[0,1] + conf_matrix[0,0])

list_qda.append(metrics_qda)


# =============================================================================
# Model 4 - Logistic Regression classifier 

gc.collect()

print(""""
      \nTraining a Logistic Regression classifier - Model 8 of 8\n
      """)

lr = pickle.load(open(owd+"/models/lr.pkl", 'rb'))

lr_pred_prob = lr.predict_proba(X_test)[:,1]

threshold_lr = 0.5

lr_pred = lr_pred_prob > threshold_lr

print("Done")
del lr





# =============================================================================
# Combine all the trained models by voting 

list_final = []

final_pred = np.array([])

print("\nEach trained model has a vote on every test observation\n")

for i in tqdm(range(len(X_test))):
    final_pred = np.append(final_pred, mode([gnb_pred[i], xgb_pred[i], lgb_pred[i],
                                             adb_pred[i], rf_pred[i], nn_pred[i]])[0].item())

metrics_final = {}

accuracy = accuracy_score(y_test, final_pred)
metrics_final['accuracy'] = accuracy
    
roc = roc_auc_score(y_test, final_pred)
metrics_final['auc_roc'] = roc
    
kappa = cohen_kappa_score(y_test, final_pred)
metrics_final['kappa'] = kappa

conf_matrix = confusion_matrix(y_test, final_pred)
metrics_final['conf_matrix'] = conf_matrix

metrics_final['sensitivity'] = conf_matrix[1,1] / (conf_matrix[1,1] + conf_matrix[1,0])
metrics_final['specificity'] = conf_matrix[0,0] / (conf_matrix[0,1] + conf_matrix[0,0])

list_final.append(metrics_final)

end_time = datetime.now()

time_taken = end_time - start_time

print("\nTotal time taken to run is %d." % time_taken)


