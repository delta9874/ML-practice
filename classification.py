#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  8 12:28:59 2019

@author: delta
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import linear_model
from sklearn import model_selection
from sklearn import metrics
from sklearn import preprocessing
from sklearn import utils


df=pd.read_csv("/home/delta/dataset/iris.csv")
df.info()
df.describe()
df[:4]
df[-4:]
df["class"].value_counts()
#######shuffling the data fr randomness
df=utils.shuffle(df,random_state=42)
df[:10]

sns.lmplot(x="sepallength",y="sepalwidth",data=df,fit_reg=False,hue="class")
sns.lmplot(x="sepallength",y="sepalwidth",data=df,hue="class")
sns.lmplot(x="petallength",y="petalwidth",data=df,fit_reg=False,hue="class")

df["ftype"]=(df["class"]=="Iris-virginica").astype(int)
df[["class","ftype"]][:5]


############################logistic regression
X=df[["sepallength","sepalwidth","petallength","petalwidth"]]
y=df["ftype"]
Xtrain,Xtest,ytrain,ytest=model_selection.train_test_split(X,y,test_size=0.15,random_state=42)
model=linear_model.LogisticRegression()
model.fit(Xtrain,ytrain)
pr_test=model.predict(Xtest)
pr_test[:8]
ytest[:5]
(ytest==pr_test).sum()/ytest.shape[0]

ytest.info()
pr_test.shape
ytest.shape

X=df[["petallength","petalwidth"]]
y=df["ftype"]
Xtrain,Xtest,ytrain,ytest=model_selection.train_test_split(X,y,test_size=0.15,random_state=42)
model=linear_model.LogisticRegression()
model.fit(Xtrain,ytrain)
pr_test=model.predict(Xtest)
pr_train=model.predict(Xtrain)
pr_test[:8]
ytest[:5]
(ytest==pr_test).sum()/ytest.shape[0]


metrics.confusion_matrix(ytest,pr_test)#sumarize the prediction total matching and unmatching case


Xtest.shape

metrics.accuracy_score(ytest,pr_test)

metrics.accuracy_score(ytrain,pr_train)

#from confusion matrix
metrics.precision_score(ytest,pr_test)
metrics.precision_score(ytrain,pr_train)
# f1 score ,and its importance
#log loss score


model.coef_#array([[ 0.08244876,  2.26298585]])
X.columns#Index(['petallength', 'petalwidth'], dtype='object')

model.intercept_# array([-3.98561086])

pr=model.predict_proba(Xtest)
pr_test[0]# 1
pr[0]#array([ 0.32419685,  0.67580315])
#logistic regression is also known as probablistics regression

#########################################confusion matrix and threshold
predicted1=[1 if p[1]>=.5 else 0 for p in pr]
metrics.confusion_matrix(ytest,predicted1)
metrics.confusion_matrix(ytest,pr_test)

predicted1=[1 if p[1]>=.5 else 0 for p in pr]

#########################################
distances=model.decision_function(Xtest)
Xtest.shape
distances.shape






































