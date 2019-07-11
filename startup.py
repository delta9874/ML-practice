#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  5 12:39:01 2019

@author: delta
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import linear_model
from sklearn import model_selection
from sklearn import metrics
df1=pd.read_csv("/home/delta/dataset/startups.csv")
df.info()
df.describe()
df.head()
df1.rename(columns={"R&D Spend": "rd", "Administration": "adm","Marketing Spend":"ms","State":"st"}, inplace=True)
df.head()
df.st.value_counts()
df[df.rd==0]
df[df.ms==0]
df.rd.describe()
df.ms.describe()
sns.boxplot(y="ms",x="st",data=df)
sns.boxplot(y="rd",x="st",data=df)
sns.boxplot(y="Profit",x="st",data=df)
##########preprocessing replaceing particular st ms avg in nan of that state

st_group=df1.groupby("st")
type(st_group)
st_group.mean()
st_group.describe()

rd_grp=st_group["rd"]
rd_grp.mean()
df1.groupby("st")["rd"].mean()

rd_dict=dict(df1.groupby("st")["rd"].mean())
rd_dict
ms_dict=dict(df.groupby("st")["ms"].mean())
ms_dict["Florida"]
df.drop(47,inplace=True)

for index,record in df.iterrows():
    if record["rd"]==0 :
        df["rd"][index]=rd_dict[record["st"]]
    if record["ms"]==0 :
        df["ms"][index]=ms_dict[record["st"]]
        
df.describe()        

 sns.violinplot(y="Profit",data=df)
sns.violinplot(y="Profit",x="st",inner="quart",data=df)  



sns.pairplot(df[["rd","ms","adm","Profit"]])
df.corr()["Profit"].sort_values(ascending=False)
df.corr()
df[["rd","ms","adm"]]


#################training and testing




X=df[["rd","ms","adm"]]

y=df["Profit"]
############################new feature introduce in the system
df["perdm"]=df["adm"]/(df.rd+df.adm+df.ms)
df[["rd","ms","adm","Profit"]].corr()
sns.lmplot(x="perdm",y="Profit",data=df)



df.Profit.describe()


Xtrain,Xtest,ytrain,ytest=model_selection.train_test_split(X,y,test_size=0.15,random_state=42)
Xtrain.shape[0]
Xtrain.info()
Xtest.info()
model=linear_model.LinearRegression()
model.fit(Xtrain,ytrain)


predict_train=model.predict(Xtrain)
np.sqrt(np.mean((ytrain-predict_train)**2))#14300.984842307045
##mean square error#root#RMSE
predict_test=model.predict(Xtest)
np.sqrt(np.mean((ytest - predict_test)**2))#4192.7560260093

predict_test=model.predict(Xtest)
np.mean((ytest - predict_test)**2)

metrics.mean_squared_error(ytest,predict_test)
model.score(Xtrain,ytrain)##coeficient of determination 
#0.86596132440490337
model.score(Xtest,ytest)
#0.80142643730632446

##################now with the new feature

Xtrain.drop("adm",axis=1,inplace=True)
Xtest.drop("adm",axis=1,inplace=True)
df1=df.copy()
X=df1[["rd","ms","perdm"]]
Y=df1[["Profit"]]

Xtrain,Xtest,ytrain,ytest=model_selection.train_test_split(X,y,test_size=0.15,random_state=42)
Xtrain.shape[0]
Xtrain.info()
Xtest.info()
model=linear_model.LinearRegression()
model.fit(Xtrain,ytrain)


predict_train=model.predict(Xtrain)
np.sqrt(np.mean((ytrain-predict_train)**2))#14280.792486303071
##mean square error#root#RMSE
predict_test=model.predict(Xtest)
np.sqrt(np.mean((ytest - predict_test)**2))#15705.110143342852

predict_test=model.predict(Xtest)
np.mean((ytest - predict_test)**2)

metrics.mean_squared_error(ytest,predict_test)
model.score(Xtrain,ytrain)##coeficient of determination 
# 0.86633957050708321
model.score(Xtest,ytest)
#0.75685243412673564
##########################################################

#Taking state in the picture

########################################################

### one hot incoding 
encdf=pd.get_dummies(df,columns=["st"],prefix="st")#in columns pass the columns which are categorical in a list
encdf.info()
df[:1]
encdf[:1]
encdf[["st_California","st_Florida","st_New York"]][:1]
#   st_California  st_Florida  st_New York
#0              0           0            1#one hot encoding

np.sqrt(np.mean((ytrain-ytrain.mean())**2))#RMSE#39061.676801828093

X=encdf[["st_California","st_Florida","st_New York"]]
y=encdf["Profit"]


Xtrain,Xtest,ytrain,ytest=model_selection.train_test_split(X,y,test_size=0.15,random_state=42)
Xtrain.shape[0]
Xtrain.info()
Xtest.info()
model=linear_model.LinearRegression()
model.fit(Xtrain,ytrain)


predict_train=model.predict(Xtrain)
np.sqrt(np.mean((ytrain-predict_train)**2))
###14280.792486303071
model.coef_
##array([  6.84707236e-01,   7.65223067e-02,  -1.17323564e+04])
metrics.mean_squared_error(ytest,predict_test)
model.score(Xtrain,ytrain)##coeficient of determination 
#0.86633957050708321
model.score(Xtest,ytest)
#0.75685243412673564


#rd ,peradm ,st test







































