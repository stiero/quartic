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
from data_preprocessing import process_data


start_time = datetime.now()
owd = os.getcwd()

# =============================================================================
# Load required data 


train, response, test, test_ids = process_data("data_train.csv", "data_test.csv", 
                                               pca=False, scale=True, imbalance_corr=True)




# =============================================================================
# Model 1 - Random Forest Classifier

gc.collect()

print("\nTraining Random Forest classifier - Model 1 of 8\n")


list_rf = []

rf = RandomForestClassifier(n_estimators = 500, random_state = 50, verbose = 1,
                                       n_jobs = -1, oob_score=True)

rf.fit(train, response)

feature_imp = list(rf.feature_importances_)

feature_importance_values = pd.DataFrame({"importance": sorted(feature_imp, reverse = True)},
                                          index=train.columns)

print("/nAs per the Random Forest Classifier, the 50 most important features are")
feature_importance_values.iloc[:51,]

rf_pred_prob = rf.predict_proba(test)[:,1]

threshold_rf = 0.4

rf_pred = rf_pred_prob > threshold_rf

print("Done")
del rf




# =============================================================================
# Model 2 - Gaussian Naive Bayes classifier

gc.collect()

print("\nTraining Gaussian Naive Bayes classifier - Model 2 of 8\n")

gnb = GaussianNB()

gnb.fit(train, response)

gnb_pred_prob = gnb.predict_proba(test)[:,1]

threshold_gnb = 0.3

gnb_pred = gnb_pred_prob > threshold_gnb

print("Done")
del gnb




# =============================================================================
# Model 3 - XGBoost classifier 

gc.collect()

print("\nTraining XG Boost classifier - Model 3 of 8\n")

dtrain = xgb.DMatrix(train, label=response)

dtest = xgb.DMatrix(test)


params = {'max_depth': 2, 'eta': 0.5, 'silent': 1, 'objective': 'binary:logistic',
          'nthread': 4, 'eval_metric': 'auc', 'colsample_bytree': 0.8, 'subsample': 0.8, 
          'scale_pos_weight': 1, 'gamma': 200, 'learning_rate': 0.02}


num_rounds = 500

threshold_xgb = 0.5

bst = xgb.train(params, dtrain, num_rounds)

xgb_pred_prob = bst.predict(dtest)

xgb_pred = xgb_pred_prob > threshold_xgb

print("Done")
del bst




# =============================================================================
# Model 4 - Logistic Regression classifier 

gc.collect()

print(""""
      =============================================================================
      \nTraining a Logistic Regression classifier - Model 4 of 8\n
      """)

lr = pickle.load(open(owd+"/models/lr.pkl", 'rb'))

lr_pred_prob = lr.predict_proba(test)[:,1]

threshold_lr = 0.5

lr_pred = lr_pred_prob > threshold_lr

print("Done")
del lr




# =============================================================================
# Model 6 - Light Gradient Boosting Machine Classifier 

gc.collect()

print("\nTraining LGBM classifier - Model 5 of 8\n")

lgb = LGBMClassifier(n_estimators=500, objective='binary', class_weight='balanced',
                     learning_rate=0.005, reg_alpha=0.5, reg_lambda=0.3, subsample=0.8,
                     n_jobs=-1, random_state=50)

lgb.fit(train, response, eval_metric='auc', verbose=True)

#lgb_pred = lgb.predict(X_test)
lgb_pred_prob = lgb.predict_proba(test)[:,1]

threshold_lgb = 0.4

lgb_pred = lgb_pred_prob > threshold_lgb
del lgb



# =============================================================================
# Model 6 - Adaptive Boosting Classifier

gc.collect()

print("\nTraining AdaBoost classifier - Model 6 of 8\n")

list_adb = []

adb = AdaBoostClassifier(n_estimators = 500, learning_rate = 0.76, algorithm = 'SAMME.R')

adb.fit(train, response)

adb_pred_prob = adb.predict_proba(test)[:,1]

threshold_adb = 0.4

adb_pred = adb_pred_prob > threshold_adb
del adb




# =============================================================================
# Model 6 - Multilayer Perceptron Neural Network Classifier

gc.collect()

print("\nTraining MLP classifier - Model 7 of 8\n")

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


model.add(Dense(1, activation = 'sigmoid'))


model.compile(loss='binary_crossentropy', optimizer = 'adam', 
              metrics = ['accuracy'])

model.fit(train, response, epochs = 10, batch_size = 32)

nn_pred_prob = model.predict(test)

threshold_nn = 0.4

nn_pred = nn_pred_prob > threshold_nn
del model



# =============================================================================
# Model 8 - Discriminant analysis

gc.collect()

list_qda = []

print("\nTraining Quadratic discriminant analysis classifier - Model 8 of 8\n")


qda = QuadraticDiscriminantAnalysis(reg_param=0.0001, tol=0.0001)

qda.fit(train, response)

qda_pred_prob = qda.predict_proba(test)[:,1]

threshold_qda = 0.0001

qda_pred = qda_pred_prob > threshold_qda
del qda



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


