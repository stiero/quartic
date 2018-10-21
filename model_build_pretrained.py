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
from sklearn.linear_model import LogisticRegression
from sklearn.externals import joblib

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


from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout

#Custom file for data loading and preprocessing
from data_preprocessing import process_data, data_split


start_time = datetime.now()
owd = os.getcwd()

print("\nUses pretrained models except for Random Forest\n")

# =============================================================================
# Load required data 


train, response, test, test_ids = process_data("data_train.csv", "data_test.csv", 
                                               pca=False, scale=True)




# =============================================================================
# Model 1 - Random Forest classifier 

gc.collect()

print(""""
      =============================================================================
      \nTraining a Random Forest classifier - Model 1 of 7\n
      """)

try:
    rf = joblib.load(owd+"/models/rf.joblib")
    
except FileNotFoundError:
    rf = RandomForestClassifier(n_estimators = 500, random_state = 50, verbose = 1,
                                       n_jobs = -1, oob_score=True)


rf.fit(train, response)

feature_importance_values = rf.feature_importances_

rf_pred_prob = rf.predict_proba(test)[:,1]

threshold_rf = 0.4

rf_pred = rf_pred_prob > threshold_rf

#joblib.dump(rf, "rf.joblib", compress=6)

print("Done")
del rf





# =============================================================================
# Model 2 - Gaussian Naive Bayes classifier

gc.collect()

print("""
      =============================================================================
      \nTraining a Gaussian Naive Bayes classifier - Model 2 of 7\n
      """)

gnb = pickle.load(open(owd+"/models/gnb.pkl", 'rb'))

gnb_pred_prob = gnb.predict_proba(test)[:,1]

threshold_gnb = 0.3

gnb_pred = gnb_pred_prob > threshold_gnb
print("Done")

del gnb
 




# =============================================================================
# Model 3 - XGBoost classifier 

gc.collect()

print(""""
      =============================================================================
      \nTraining an XGBoost classifier - Model 3 of 7\n
      """)


dtrain = xgb.DMatrix(train, label=response)
dtest = xgb.DMatrix(test)

params = {'max_depth': 2, 'eta': 0.5, 'silent': 0, 'objective': 'binary:logistic',
          'nthread': 4, 'eval_metric': 'auc', 'colsample_bytree': 0.8, 'subsample': 0.8, 
          'scale_pos_weight': 1, 'gamma': 200, 'learning_rate': 0.02}


evallist = [(dtest, 'eval'), (dtrain, 'train')]

num_rounds = 500


bst = pickle.load(open(owd+"/models/xgb.pkl", 'rb'))
xgb_pred_prob = bst.predict(dtest)

threshold_xgb = 0.4


xgb_pred = xgb_pred_prob > threshold_xgb

print("Done")
del bst





# =============================================================================
# Model 4 - Logistic Regression classifier 

gc.collect()

print(""""
      =============================================================================
      \nTraining a Logistic Regression classifier - Model 4 of 7\n
      """)

lr = pickle.load(open(owd+"/models/lr.pkl", 'rb'))

lr_pred_prob = lr.predict_proba(test)[:,1]

threshold_lr = 0.5

lr_pred = lr_pred_prob > threshold_lr

print("Done")
del lr




# =============================================================================
# Model 5 - Discriminant analysis

gc.collect()

list_qda = []

qda = pickle.load(open(owd+"/models/qda.pkl", 'rb'))

qda.fit(train, response)

qda_pred_prob = qda.predict_proba(test)[:,1]

threshold_qda = 0.0001

qda_pred = qda_pred_prob > threshold_qda

print("Done")
del qda




# =============================================================================
# Model 5 - Light Gradient Boosting Machine Classifier 

gc.collect()

print(""""
      =============================================================================
      \nTraining a Light Gradient Boosting classifier - Model 5 of 7\n
      """)
lgb = pickle.load(open(owd+"/models/lgb.pkl", 'rb'))

lgb_pred_prob = lgb.predict_proba(test)[:,1]

threshold_lgb = 0.4

lgb_pred = lgb_pred_prob > threshold_lgb

print("Done")
del lgb




# =============================================================================
# Model 6 - Adaptive Boosting Classifier

gc.collect()

print(""""
      =============================================================================
      \nTraining a Adaptive Boosting classifier - Model 6 of 7\n
      """)

adb = pickle.load(open(owd+"/models/adb.pkl", 'rb'))

adb_pred_prob = adb.predict_proba(test)[:,1]

threshold_adb = 0.4

adb_pred = adb_pred_prob > threshold_adb

print("Done")
del adb




# =============================================================================
# Model 7 - Multilayer Perceptron Neural Network Classifier

gc.collect()

print(""""
      =============================================================================
      \nTraining a Multilayer Perceptron classifier - Model 7 of 7\n
      """)

model = load_model(owd+"/models/nn.h5")

nn_pred_prob = model.predict(test)

threshold_nn = 0.4

nn_pred = nn_pred_prob > threshold_nn

print("Done")
del model




# =============================================================================
# Combine all the trained models by voting 

final_pred = np.array([])

print("\nEach trained model has a vote on every test observation. The majority prediction wins\n")

for i in tqdm(range(len(test))):
    final_pred = np.append(final_pred, mode([gnb_pred[i], xgb_pred[i], lgb_pred[i],
                                             nn_pred[i], adb_pred[i], rf_pred[i]])[0].item())

final_pred = pd.DataFrame({"id": test_ids, "target": final_pred})


final_pred.to_csv("voting_ensemble_predictions.csv", index=True) 
 
print("\nOutput file written as a csv\n")

end_time = datetime.now()

time_taken = end_time - start_time

print("\nTotal time taken to run is %s." % str(time_taken))




