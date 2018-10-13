#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 23 13:56:17 2018

@author: tauro
"""
import numpy as np
import pandas as pd

from sklearn.preprocessing import LabelEncoder

import warnings
warnings.filterwarnings('ignore')

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler

from sklearn.metrics import accuracy_score, roc_auc_score, cohen_kappa_score, confusion_matrix

from sklearn.model_selection import train_test_split

from sklearn.utils import resample

from tqdm import tqdm
tqdm.pandas(desc="progress-bar")


train = pd.read_csv("data_train.csv")
test = pd.read_csv("data_test.csv")

response = train['target']
train_ids = train['id']
test_ids = test['id']

del train['target']
del train['id']
del test['id']

train_cols = list(train)
test_cols = list(test)

response.value_counts()
#There is class imbalance

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

train.dtypes

def cat_cols_to_object(df):
    cat_cols = [element for element in list(df) if 'cat' in element]
    
    for col in cat_cols:
        df[col] = df[col].astype(object)
        
    return df


train = cat_cols_to_object(train)
test = cat_cols_to_object(test)

#Number of unique classes in each object predictor
train.select_dtypes('object').apply(pd.Series.nunique, axis = 0).sum()
test.select_dtypes('object').apply(pd.Series.nunique, axis = 0)

missing_cats_train = missing_values(train.filter(regex='cat'))
missing_cats_test = missing_values(test.filter(regex='cat'))

#train_new = pd.DataFrame(index=train['id'])
train_cats_encoded = pd.DataFrame()
test_cats_encoded = pd.DataFrame()

#Encoding categorical variables
le = LabelEncoder()
sc = StandardScaler()
le_count = 0
ohe_count = 0

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
        
        #To suppress type conversion warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            
            sc.fit(np.array(train[col]).reshape(-1,1))
        
            train[col] = sc.transform(np.array(train[col]).reshape(-1,1))
            test[col] = sc.transform(np.array(test[col]).reshape(-1,1))
            
        
print('%d categorical columns were label encoded' %le_count)
print('%d categorical columns were one-hot encoded' %ohe_count)

train = pd.concat([train, train_cats_encoded], axis = 1)
test = pd.concat([test, test_cats_encoded], axis = 1)


#Umputing missing int/float values
from sklearn.preprocessing import Imputer

def imputer(train):
    imputer = Imputer(strategy = 'median')
    
    #train['num18'] = imputer.fit_transform(np.array(train['num18']).reshape(-1,1))
    #test['num18'] = imputer.transform(np.array(test['num18']).reshape(-1,1))
    
    #train['num22'] = imputer.fit_transform(np.array(train['num22']).reshape(-1,1))
    #test['num22'] = imputer.transform(np.array(test['num22']).reshape(-1,1))
    
    #train['num19'] = imputer.fit_transform(np.array(train['num19']).reshape(-1,1))
    #test['num19'] = imputer.transform(np.array(test['num19']).reshape(-1,1))
    
    #train['num20'] = imputer.fit_transform(np.array(train['num20']).reshape(-1,1))
    #test['num20'] = imputer.transform(np.array(test['num20']).reshape(-1,1))
    
    cols_train = [str(col) for col in train.columns]
    
    imp_count = 0
    
    for col in tqdm(cols_train):
        if "num" in col:
            train[col] = imputer.fit_transform(np.array(train[col]).reshape(-1,1))
            test[col] = imputer.transform(np.array(test[col]).reshape(-1,1))
            imp_count += 1
            
    print('Missing values in %d columns were imputed' %imp_count)
    
    train['target'] = response
    
    return train, test

train, test = imputer(train)



#PCA

def compute_pca(df, n_comps):
    
    from sklearn.decomposition import PCA    
    
    pca = PCA(n_components = n_comps, copy=True)        
    
    df = train.copy()        
    del df['target']        
    
    df = pca.fit_transform(df)        
    df = np.column_stack((df, np.array(response)))
    
    print(pca.explained_variance_ratio_)        
    print(sum(pca.explained_variance_ratio_))
    
    return pd.DataFrame(df)

#train = compute_pca(train, 50)


# Train test split

try:
    del train['target']
except KeyError:
    train['target'] = response

X_train, X_test, y_train, y_test = train_test_split(train, response,
                                                    test_size=0.1, random_state = 3536)


# Addressing class imbalance

X_train['target'] = y_train

train_maj = X_train[X_train['target'] == 0]
train_min = X_train[X_train['target'] == 1]

train_min_sampled = resample(train_min, replace = True,
                             n_samples = len(train_maj), random_state = 9868)

X_train = pd.concat([train_maj, train_min_sampled])

y_train = X_train['target']
del X_train['target']

#del X_train['target'], X_test['target']

#df_response = X_train['target']








###################################################

#Logistic regression

from sklearn.linear_model import LogisticRegression

metrics_log_reg = {}

C = 0.1

log_reg = LogisticRegression(C=C, class_weight='balanced')

log_reg.fit(X_train, y_train)

log_reg_pred = pd.Series(log_reg.predict(X_test))

metrics_log_reg['C'] = C

accuracy = accuracy_score(y_test, log_reg_pred)
metrics_log_reg['accuracy'] = accuracy
    
roc = roc_auc_score(y_test, log_reg_pred)
metrics_log_reg['auc_roc'] = roc
    
kappa = cohen_kappa_score(y_test, log_reg_pred)
metrics_log_reg['kappa'] = kappa

conf_matrix = confusion_matrix(y_test, log_reg_pred)
metrics_log_reg['conf_matrix'] = conf_matrix

log_reg_pred.value_counts()



#####################################################

# Gaussian Naive Bayes

from sklearn.naive_bayes import GaussianNB

metrics_gau_nb = {}

gau_nb = GaussianNB()

gau_nb.fit(X_train, y_train)

gau_nb_pred = pd.Series(gau_nb.predict(X_test))

accuracy = accuracy_score(y_test, gau_nb_pred)
metrics_gau_nb['accuracy'] = accuracy
    
roc = roc_auc_score(y_test, gau_nb_pred)
metrics_gau_nb['auc_roc'] = roc
    
kappa = cohen_kappa_score(y_test, gau_nb_pred)
metrics_gau_nb['kappa'] = kappa

conf_matrix = confusion_matrix(y_test, gau_nb_pred)
metrics_gau_nb['conf_matrix'] = conf_matrix

gau_nb_pred.value_counts()

########################################################
# Random forest

from sklearn.ensemble import RandomForestClassifier

metrics_rf = {}

rf = RandomForestClassifier(n_estimators = 100, random_state = 50, verbose = 1,
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

pd.Series(rf_pred).value_counts()


##############################################

# Light GBM

from lightgbm import LGBMClassifier

list_lgb = []

metrics_lgb = {}

lgb = LGBMClassifier(n_estimators=5000, objective='binary', class_weight='balanced',
                     learning_rate=0.005, reg_alpha=0.5, reg_lambda=0.3, subsample=0.8,
                     n_jobs=-1, random_state=50)

lgb.fit(X_train, y_train, eval_metric='auc', verbose=True)

#lgb_pred = lgb.predict(X_test)
lgb_pred_prob = lgb.predict_proba(X_test)[:,1]

threshold = 0.5

lgb_pred = lgb_pred_prob > threshold

accuracy = accuracy_score(y_test, lgb_pred)
metrics_lgb['accuracy'] = accuracy
    
roc = roc_auc_score(y_test, lgb_pred)
metrics_lgb['auc_roc'] = roc
    
kappa = cohen_kappa_score(y_test, lgb_pred)
metrics_lgb['kappa'] = kappa

conf_matrix = confusion_matrix(y_test, lgb_pred)
metrics_lgb['conf_matrix'] = conf_matrix

list_lgb.append(metrics_lgb)

pd.Series(lgb_pred).value_counts()


########################################

import xgboost as xgb

list_xgb = []




dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)

params = {'max_depth': 2, 'eta': 0.5, 'silent': 0, 'objective': 'binary:logistic',
          'nthread': 4, 'eval_metric': 'auc', 'colsample_bytree': 0.8, 'subsample': 0.8, 
          'scale_pos_weight': 26, 'gamma': 200, 'learning_rate': 0.02}


evallist = [(dtest, 'eval'), (dtrain, 'train')]

num_rounds = 5000


bst = xgb.train(params, dtrain, num_rounds, evallist)

bst_pred = bst.predict(dtest)

threshold = 0.5

xgb_pred = bst_pred > threshold

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

sensitivity = conf_matrix[1,1] / (conf_matrix[1,1] + conf_matrix[1,0])
specificity = conf_matrix[0,0] / (conf_matrix[0,1] + conf_matrix[0,0])

list_xgb.append(metrics_xgb)

pd.Series(xgb_pred).value_counts()

############################################

from sklearn.neighbors import KNeighborsClassifier

list_knn = []

knn = KNeighborsClassifier(n_neighbors = 5, weights='uniform', n_jobs=-1)

knn.fit(X_train, y_train)

knn_pred = knn.predict(X_test)

metrics_knn = {}

accuracy = accuracy_score(y_test, knn_pred)
metrics_knn['accuracy'] = accuracy

roc = roc_auc_score(y_test, knn_pred)
metrics_knn['auc_roc'] = roc

kappa = cohen_kappa_score(y_test, knn_pred)
metrics_knn['kappa'] = kappa

conf_matrix = confusion_matrix(y_test, knn_pred)
metrics_knn['conf_matrix'] = conf_matrix

list_knn.append(metrics_knn)

pd.Series(knn_pred).value_counts()

################################################

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

#################################################

corrs = train.corr()
s = corrs.unstack()
so = s.sort_values(kind="quicksort", ascending=False)
