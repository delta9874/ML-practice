#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  4 11:50:53 2019

@author: delta
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
#########scikit -learn
from sklearn import linear_model


df=pd.read_csv("/home/delta/dataset/bp-age.csv")
df.describe()
df.info()
df
sns.lmplot(x="age",y="bp",data=df,fit_reg=False)
sns.lmplot(x="weight",y="bp",data=df,fit_reg=False)
sns.lmplot(x="weight",y="age",data=df,fit_reg=False)
sns.distplot(df["bp"])#distribution plot plot normality#y should e normal otherwise alogo wont work
sns.pairplot(df,diag_kind="kde")#find the relation between different variables 
#correlation coefficient to check the retaion between variables  -1,0,1
df.corr()#correlation#check graph as well as corr coff in case of conflict or to make sur
sns.heatmap(df.corr(),annot=True)
#steps to proceed before jumping to algo ,dataset must not have any string and nan the proceed by ploting
#and check for outlairs too
df.isnull().sum()/df.shape[0]
# co linear====>relationship between different x variable
#linear regression is predefined class in scikit learn
###################scikit learn############################

X=df["age"].reshape(X.shape[0],1)## full matrix
#X=df["age"].values.reshape(-1,1)
X.shape
X
y=df["bp"]





model=linear_model.LinearRegression()
#if you want to use in any model of scikit learn data must be divided in x part and y part
model.fit(X,y)#X must be in matrx format
print ("intercept:",model.intercept_)
print("coef:",model.coef_)
######bp=  intercept + coef*age, bp= 58.7055 +1.4623*age
df[:2]
model.predict([[59],[52]])
###claculated :array([ 145.03611293]) ###actual =143





############################2d data
X=df[["age","weight"]]
y=df["bp"]
model=linear_model.LinearRegression()
#if you want to use in any model of scikit learn data must be divided in x part and y part
model.fit(X,y)#X must be in matrx format

model.predict([[59,184]])
#####calculated :array([ 143.43166173]) ###actual =143
model.coef_
model.intercept_

###
























