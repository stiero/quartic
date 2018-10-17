#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  9 15:38:55 2018

@author: ishippoml
"""



###########################################################################

from statsmodels.stats.outliers_influence import variance_inflation_factor    

def calculate_vif_(X, thresh=5.0):
    variables = list(range(X.shape[1]))
    dropped = True
    while dropped:
        dropped = False
        vif = [variance_inflation_factor(X.iloc[:, variables].values, ix)
               for ix in range(X.iloc[:, variables].shape[1])]

        maxloc = vif.index(max(vif))
        if max(vif) > thresh:
            print('dropping \'' + X.iloc[:, variables].columns[maxloc] +
                  '\' at index: ' + str(maxloc))
            del variables[maxloc]
            dropped = True

    print('Remaining variables:')
    print(X.columns[variables])
    return X.iloc[:, variables]

vif = calculate_vif_(train, thresh=5.0)


"""dropping 'num7' at index: 6
dropping 'num3' at index: 2
dropping 'num20' at index: 17
dropping 'num22' at index: 18
dropping 'num21' at index: 17
dropping 'cat13_1' at index: 93
dropping 'cat9_10' at index: 79
dropping 'num11' at index: 8
dropping 'der8' at index: 24
dropping 'cat4_11.0' at index: 55
dropping 'der6' at index: 22
dropping 'num23' at index: 16
dropping 'cat9_11' at index: 74
dropping 'num19' at index: 15
dropping 'cat9_1' at index: 64
dropping 'num18' at index: 14
dropping 'cat10_1.0' at index: 77
dropping 'cat14_104' at index: 185"""