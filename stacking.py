#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 12 14:48:43 2018

@author: tauro
"""

from sklearn.model_selection import StratifiedKFold    

from sklearn.naive_bayes import GaussianNB

from sklearn.ensemble import RandomForestClassifier

from lightgbm import LGBMClassifier

from xgboost import XGBClassifier



def stacking(model, train, response, n_fold):

    kfold = StratifiedKFold(n_splits=n_fold, shuffle=True, random_state=1986)
    
    list_gnb = []
    
    #train_new = np.empty((0,1), float)
    train_new = pd.DataFrame()
    
    for tr, te in kfold.split(train, response):
        train_cv = train.iloc[tr]
        test_cv = train.iloc[te]
        
        response_train = response.iloc[tr]
        response_test = response.iloc[te]
        
        metrics_model = {}
        
        model.fit(train_cv, response_train)
        
        model_pred = pd.DataFrame({str(model): model.predict(test_cv)}, index=te)
        #model_pred_test = pd.Series(model.predict(test))
        
        accuracy = accuracy_score(response_test, model_pred)
        metrics_model['accuracy'] = accuracy
            
        roc = roc_auc_score(response_test, model_pred)
        metrics_model['auc_roc'] = roc
            
        kappa = cohen_kappa_score(response_test, model_pred)
        metrics_model['kappa'] = kappa
        
        conf_matrix = confusion_matrix(response_test, model_pred)
        metrics_model['conf_matrix'] = conf_matrix
        
        list_gnb.append(metrics_model)
        
        train_new = pd.concat([train_new, model_pred], axis=0)
         
    return train_new, model_pred_test



params = {'max_depth': 2, 'eta': 0.5, 'silent': 0, 'objective': 'binary:logistic',
          'nthread': 4, 'eval_metric': 'auc', 'colsample_bytree': 0.8, 'subsample': 0.8, 
          'scale_pos_weight': 26, 'gamma': 200, 'learning_rate': 0.02}



gnb = GaussianNB()

rf = RandomForestClassifier(n_estimators = 2000, random_state = 50, verbose = 1,
                                       n_jobs = -1, oob_score=True)

lgb = LGBMClassifier(n_estimators=2000, objective='binary', class_weight='balanced',
                     learning_rate=0.005, reg_alpha=0.5, reg_lambda=0.3, subsample=0.8,
                     n_jobs=-1, random_state=50)

xgb = XGBClassifier(n_estimators=2000, **params)


train_gnb, test_gnb = stacking(gnb, train = train, response = response, n_fold=10)
train = pd.concat([train, train_gnb], axis = 1)
test = pd.concat([test, test_gnb], axis = 1)

train_rf, test_rf = stacking(rf, train = train, response = response, n_fold = 10)
train = pd.concat([train, train_rf], axis = 1)
test = pd.concat([test, test_rf], axis = 1)

train_lgb, test_lgb = stacking(lgb, train = train, response = response, n_fold = 10)
train = pd.concat([train, train_lgb], axis = 1)
test = pd.concat([test, test_lgb], axis = 1)





    

    
