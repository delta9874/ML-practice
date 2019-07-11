#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  4 14:07:43 2019

@author: delta
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import linear_model
from sklearn import model_selection
from sklearn import metrics
##########################################
arr2d=np.array([[10,20],[40,50]])#2d array to 1d array
arr2d
arr1d=arr2d.flatten()
arr1d




################ols =ordinary Least Squares
df=pd.read_csv("/home/delta/dataset/advertising.csv")
df.info()
df.head()
df.describe()
df[170:188]
df.drop("srno",axis=1,inplace=True)

##################boxploting 
fig,axes=plt.subplots(nrows=2,ncols=3,figsize=(6,6))
axes1=axes.flatten()
index=0
for col in df.columns.values:
    sns.boxplot(y=col,data=df,ax=axes1[index])
    index+=1
plt.tight_layout()
#####################violinplot
fig,axes=plt.subplots(nrows=2,ncols=3,figsize=(6,6))
axes1=axes.flatten()
index=0
for col in df.columns.values:
    sns.violinplot(y=col,data=df,ax=axes1[index])
    index+=1
plt.tight_layout()


sns.distplot(df["sales"])
sns.distplot(df["radio"])
sns.distplot(np.log((df["radio"]+0.1)))
df.corr()["sales"].sort_values(ascending=False)
sns.heatmap(df.corr(),annot=True)#co relation between themself
###############

X=df.drop("sales",axis=1)
y=df["sales"]

Xtrain,Xtest,ytrain,ytest=model_selection.train_test_split(X,y,test_size=0.15,random_state=42)
Xtrain.shape[0]
Xtrain.info()
Xtest.info()
model=linear_model.LinearRegression()
model.fit(Xtrain,ytrain)
predict_train=model.predict(Xtrain)
np.mean((ytrain-predict_train)**2)##mean square error
predict_test=model.predict(Xtest)
np.mean((ytest - predict_test)**2)

metrics.mean_squared_error(ytest,predict_test)
model.score(Xtrain,ytrain)##coeficient of determination 
##adjusted R square
#how our X is defining  variability of y
###how much change in y is affecting x
model.score(Xtest,ytest)

#####MSE ==> perfromance metrics


X=df[["TV","radio"]]
y=df["sales"]

Xtrain,Xtest,ytrain,ytest=model_selection.train_test_split(X,y,test_size=0.15,random_state=42)
Xtrain.shape[0]
Xtrain.info()
Xtest.info()
model=linear_model.LinearRegression()
model.fit(Xtrain,ytrain)


predict_train=model.predict(Xtrain)
np.mean((ytrain-predict_train)**2)##mean square error
predict_test=model.predict(Xtest)
np.mean((ytest - predict_test)**2)

predict_test=model.predict(Xtest)
np.mean((ytest - predict_test)**2)

metrics.mean_squared_error(ytest,predict_test)
model.score(Xtrain,ytrain)##coeficient of determination 
##adjusted R square
#how our X is defining  variability of y
###how much change in y is affecting x
model.score(Xtest,ytest)


df["radio_log"]=pd.Series(np.log(df["radio"]+.01))

df1=pd.read_csv("/home/delta/dataset/housing.csv")
df1.shape[0]
df1.info()
#abnormal scaling ,categorical, contious,out lairs,nan % in respective columns , normality, which columns influence the target most
#
for col in df.columns :
    sns.distplot(df[col])

####################################
from sklearn import preprocessing
X_scaled = preprocessing.scale(df)
sns.countplot(x=)


###############3abnscling
df["alat"] = (df.long - df.long.min() )/(df.long.max()-df.long.min())
df["alat"].describe()
sns.distplot(df.alat)
#########################################
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import linear_model
from sklearn import model_selection
from sklearn import metrics
from sklearn import preprocessing


df=pd.read_csv("/home/delta/dataset/advertising.csv")
df.info()
df.head()
df.describe()
df[170:188]
df.drop("srno",axis=1,inplace=True)

X=df["TV"].values.reshape(-1,1)
y=df["sales"]

Xtrain,Xtest,ytrain,ytest=model_selection.train_test_split(X,y,test_size=0.15,random_state=42)
Xtrain.shape[0]
Xtrain.info()
Xtest.info()
model=linear_model.LinearRegression()
model.fit(Xtrain,ytrain)


predict_train=model.predict(Xtrain)
np.mean((ytrain-predict_train)**2)##mean square error
predict_test=model.predict(Xtest)
np.mean((ytest - predict_test)**2)

predict_test=model.predict(Xtest)
np.mean((ytest - predict_test)**2)

metrics.mean_squared_error(ytest,predict_test)
model.score(Xtrain,ytrain)##coeficient of determination 
##adjusted R square
#how our X is defining  variability of y
###how much change in y is affecting x
model.score(Xtest,ytest)
sns.lmplot(x="TV",y="sales",data=df,fit_reg=False)
sns.lmplot(x="radio",y="sales",data=df,fit_reg=False)
sns.lmplot(x="newspaper",y="sales",data=df,fit_reg=False)
model.coef_
#array([ 0.04625968])

model.intercept_
#7.175905817692672

sales =0.046* TV+ 7.175#this gives the idea tv is not important parameter for sales
#as unit increase in TV only 0.046 is affect in sales


####################tv**2

df["TVsq"]=df.TV**2

X=df[["TV","TVsq"]]
y=df["sales"]

Xtrain,Xtest,ytrain,ytest=model_selection.train_test_split(X,y,test_size=0.15,random_state=42)
Xtrain.shape[0]
Xtrain.info()
Xtest.info()
model=linear_model.LinearRegression()
model.fit(Xtrain,ytrain)


predict_train=model.predict(Xtrain)
np.mean((ytrain-predict_train)**2)##mean square error
predict_test=model.predict(Xtest)
np.mean((ytest - predict_test)**2)

predict_test=model.predict(Xtest)
np.mean((ytest - predict_test)**2)

metrics.mean_squared_error(ytest,predict_test)
model.score(Xtrain,ytrain)##coeficient of determination 
##adjusted R square
#how our X is defining  variability of y
###how much change in y is affecting x
model.score(Xtest,ytest)
sns.lmplot(x="TV",y="sales",data=df,fit_reg=False)
sns.lmplot(x="radio",y="sales",data=df,fit_reg=False)
sns.lmplot(x="newspaper",y="sales",data=df,fit_reg=False)
model.coef_

array([  6.65534870e-02,  -6.94129383e-05])
model.intercept_
6.1952865575406264

formula
sales=6.66*TV -6.94*TVsq +6.2

#bias ,irreduceable error,variance #controling parameter
#####preprocessing
poly=preprocessing.PolynomialFeatures(degree=2,include_bias=False)
df=pd.read_csv("/home/delta/dataset/advertising.csv")
df.drop("srno",axis=1,inplace=True)
X=df[["TV","radio"]]
y=df["sales"]
X_poly=poly.fit_transform(X)
X[:1]
X_poly.shape
X_poly[:1]#array([[  2.30100000e+02,   3.78000000e+01,   5.29460100e+04,
          #8.69778000e+03,   1.42884000e+03]])
poly.get_feature_names()# ['x0', 'x1', 'x0^2', 'x0 x1', 'x1^2']


Xtrain,Xtest,ytrain,ytest=model_selection.train_test_split(X_poly,y,test_size=0.15,random_state=42)
Xtrain.shape[0]
Xtrain.info()
Xtest.info()
model=linear_model.LinearRegression()
model.fit(Xtrain,ytrain)


predict_train=model.predict(Xtrain)
np.mean((ytrain-predict_train)**2)##mean square error
predict_test=model.predict(Xtest)
np.mean((ytest - predict_test)**2)

predict_test=model.predict(Xtest)
np.mean((ytest - predict_test)**2)

metrics.mean_squared_error(ytest,predict_test)
model.score(Xtrain,ytrain)##coeficient of determination 
##adjusted R square
#how our X is defining  variability of y
###how much change in y is affecting x
model.score(Xtest,ytest)

































