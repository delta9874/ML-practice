#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 13 12:25:58 2019

@author: delta
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import linear_model
from sklearn import model_selection
from sklearn import metrics
from sklearn import ensemble
from datetime import datetime
from sklearn.preprocessing import StandardScaler 
from sklearn import utils
from sklearn import tree

from sklearn import naive_bayes
# =============================================================================
# 
# 
# Naive bayes
# 1>GaussionNB:- continous
# 2>BernoulliNB:- if only two values
# 3>MultinomialNB:-discrete variable but more than two values
# =============================================================================

df=pd.read_csv("/home/delta/dataset/iris.csv")
df[:5]
#all X are continous
##GausianNB
df=utils.shuffle(df)
df["type"]=(df["class"]=="Iris-setosa").astype(np.int)
df.drop("class",axis=1,inplace=True)

X=df.drop("type",axis=1)
y=df["type"]

X.info()

Xtrain,Xtest,ytrain,ytest=model_selection.train_test_split(X,y,test_size=0.15,random_state=42)  

model=naive_bayes.GaussianNB()
model.fit(Xtrain,ytrain)
prediction=model.predict(Xtest)

X_pred=model.predict(Xtrain)
print("confusion:",metrics.confusion_matrix(ytest,prediction))

printresult(ytest,prediction)
# =============================================================================
# [[16  0]
#  [ 0  7]]
# accuracy : 1.0000
# precision : 1.0000
# recall : 1.0000
# f1-score : 1.0000
# AUC : 1.0000
# =============================================================================
printresult(ytrain,X_pred)
# =============================================================================
# [[84  0]
#  [ 0 43]]
# accuracy : 1.0000
# precision : 1.0000
# recall : 1.0000
# f1-score : 1.0000
# AUC : 1.0000
# =============================================================================

def printresult(actual,predicted):
    confmatrix=metrics.confusion_matrix(actual,predicted)
    accscore=metrics.accuracy_score(actual,predicted)
    precscore=metrics.precision_score(actual,predicted)
    recscore=metrics.recall_score(actual,predicted)
    print(confmatrix)
    print("accuracy : {:.4f}".format(accscore))
    print("precision : {:.4f}".format(precscore))
    print("recall : {:.4f}".format(recscore))
    print("f1-score : {:.4f}".format(metrics.f1_score(actual,predicted)))
    print("AUC : {:.4f}".format(metrics.roc_auc_score(actual,predicted)))

#viaf -multicolinarity 
    

probs=model.predict_proba(Xtest)
probs


##multiclass

X=df.drop(["type","class"],axis=1)

df["type"]=df["class"].replace({"Iris-setosa":1,"Iris-versicolor":2,"Iris-virginica":3})


y=df["type"]

X.info()

Xtrain,Xtest,ytrain,ytest=model_selection.train_test_split(X,y,test_size=0.15,random_state=42)  

model=naive_bayes.GaussianNB()
model.fit(Xtrain,ytrain)
prediction=model.predict(Xtest)

X_pred=model.predict(Xtrain)
print("confusion:",metrics.confusion_matrix(ytest,prediction))

# =============================================================================
# confusion: [[8 0 0]
#  [0 9 0]
#  [0 0 6]]
# =============================================================================

for a,b,c in model.predict_proba(Xtest):
    if max(a,b,c)==a:
        print ("Iris-setosa")
    if max(a,b,c)==b:
        print ("Iris-versicolor")
    else:
        print ("Iris-virginca")







