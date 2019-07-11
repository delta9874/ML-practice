#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 12 14:38:26 2019

@author: delta
"""


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import linear_model
from sklearn import model_selection
from sklearn import metrics

from datetime import datetime
from sklearn.preprocessing import StandardScaler 

from sklearn import tree
from sklearn import ensemble


########random forest :multiple decision tree
#####boot straping:process of reducing variance ,basically resampling with replacement
X={x1,x2,........xn},n=100
300 samples of size N with replacement
bootstarp sample
XB1,XB2,........XB300

var(300 means)   = sigma^2/3009

df=pd.read_csv("/home/delta/dataset/pima_diabetics.csv")

X=df.drop("class",axis=1)
y=df["class"]

Xtrain,Xtest,ytrain,ytest=model_selection.train_test_split(X,y,test_size=0.15,random_state=42)  
rf=ensemble.RandomForestClassifier(n_estimators=300,criterion="entropy",max_depth=7)
rf.fit(X,y)
rf.fit(Xtrain,ytrain)
prediction=rf.predict(Xtest)
printresult(ytest,prediction)
# =============================================================================
# [[60 16]
#  [12 28]]
# accuracy : 0.7586
# precision : 0.6364
# recall : 0.7000
# f1-score : 0.6667
# AUC : 0.7447
# =============================================================================

















