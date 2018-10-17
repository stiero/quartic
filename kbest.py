#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 10 20:39:45 2018

@author: tauro
"""

# To select 'k' best features automatically 

from sklearn.feature_selection import SelectKBest, chi2, f_classif

kbest = SelectKBest(score_func=f_classif, k=100)

kbest.fit(train, response)

kbest_train = kbest.transform(train)
kbest_test = kbest.transform(test)

