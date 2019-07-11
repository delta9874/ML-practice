#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 13 15:05:49 2019

@author: delta
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import model_selection
from sklearn import metrics
from sklearn import ensemble
from datetime import datetime
from sklearn.preprocessing import StandardScaler 
from sklearn import utils
from sklearn import neighbors
from sklearn import preprocessing
df=pd.read_csv("/home/delta/dataset/startups.csv")
df.info()

df.rename(columns={"R&D Spend":"rs","Administration":"adm","Marketing Spend":"ms"},inplace=True)

df.info()
######label encoding
state_encoder=preprocessing.LabelEncoder()
df["enc_state"]=state_encoder.fit_transform(df["State"])
df["enc_state"]
################
df.head()

df.drop("State",axis=1,inplace=True)
df.info()

X=df.drop("Profit",axis=1)
y=df["Profit"]

Xtrain,Xtest,ytrain,ytest=model_selection.train_test_split(X,y,test_size=0.15,random_state=42)
Xtrain.info()

knnmodel=neighbors.KNeighborsRegressor(n_neighbors=11)
knnmodel.fit(Xtrain,ytrain)
#fit doesnt create model ,but it create a data structure which help us to search easier
#kdtree
#balltree
#brute
#alogorithm="........."


prediction=knnmodel.predict(Xtest)
print(np.sqrt(metrics.mean_squared_error(ytest,prediction)))

X[:3]

#standard  scaling
avg=df.rs.mean()
sd=df.rs.std()
t=(df.rs-avg)/sd
t[:3]
#min max scaling
#robust scaling-used in case of outlairs,is not affected by outlairs
colnames=X.columns.values
rb_scaler=preprocessing.RobustScaler()
X_scaled=rb_scaler.fit_transform(X)
X_scaled[:3]#return a numpy array
X_Sc=pd.DataFrame(X_scaled,columns=colnames)

Xtrain,Xtest,ytrain,ytest=model_selection.train_test_split(X_Sc,y,test_size=0.15,random_state=42)
Xtrain.info()

knnmodel=neighbors.KNeighborsRegressor(n_neighbors=11)
knnmodel.fit(Xtrain,ytrain)


prediction=knnmodel.predict(Xtest)
print(np.sqrt(metrics.mean_squared_error(ytest,prediction)))















