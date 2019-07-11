#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 31 22:42:41 2019

@author: delta
"""

    
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt #ploting and visualisation
import seaborn as sns#ploting  
df=pd.read_excel("/home/delta/dataset")
df.head()
df.info()
df2=df[["University Roll No.","Name","Contact Number","Email - id"]]
df.values()
df.index.values()
exp=df2.to_csv("/home/delta/naaccse2017.csv")
df.Board.value_counts()
sns.boxplot(x="First Semester",y="Class XII Marks",data=df)
plt.xticks(rotation=90)


####################################perceptron
data=[[2.7810836,2.550537003,0],
	[1.465489372,2.362125076,0],
	[3.396561688,4.400293529,0],
	[1.38807019,1.850220317,0],
	[3.06407232,3.005305973,0],
    #[4.722441248,2.158626775,0],
   # [4.302441248,2.058626775,0],
	[7.627531214,2.759262235,1],
	[5.332441248,2.088626775,1],
    [5.112441248,2.288626775,1],
    [4.802441248,1.8888626775,1],
    [4.502441248,2.358626775,1],
	[6.922596716,1.77106367,1],
	[8.675418651,0.242068655,1],
	[7.673756466,3.508563011,1]]

testdata=[[5.5010836,1.510537003,0],
	[6.117531214,1.159262235,1],
	[5.662441248,2.088626775,1],
    [4.442441248,1.8888626775,1],
	[7.895418651,0.242068655,1],
	]

df=pd.DataFrame(data,columns=["length","width","bugtype"])
df
sns.lmplot(x="length",y="width",hue="bugtype",fit_reg=False,data=df)
sns.lmplot(x="length",y="width",hue="bugtype",data=df)
df.describe()
###################
np.random.seed(42)
w=np.random.random(3)
w
X=(np.random.random((4,2))*100).astype(np.int)
X
calculated=X.dot(w[1:])+w[0]
calculated
prediction=(calculated>=0).astype(np.int)
prediction
######################
train=df.values
train
np.random.shuffle(train)
train
xtrain=train[:,0:2]
xtrain
ytrain=train[:,-1]

fit()

orig=np.array([1,0,1,0,1,1])
yhat=np.array([1,1,1,0,1,0])
mismatched=np.nonzero(orig!=yhat)[0]
mismatched
xtrain.shape[1]
w
calculate
#########################################################


def fit(xtrain,ytrain,lr=0.1,nepoch=1000):
    w=np.random.random(xtrain.shape[1]+1)
    for epoch in range(nepoch):
       calculate=xtrain.dot(w[1:])+w[0]
       yhat=(calculate>=0).astype(np.int)
       incorrect=np.nonzero(ytrain!=yhat)[0]
       if incorrect.shape[0]==0:
           break
       #i=np.random.choice(incorrect)
       #x=xtrain[i:]#only one record is chosen randomly 
       #update=lr*(ytrain[i]-yhat[i])
       update=lr*(ytrain-yhat)
       for i,r in enumerate(xtrain):
           #r=[2,4]
           #update[i]=10
           #update[i]*r=10*[2,4]=20,40
           w[1:]+=update[i]*r
           w[0]+=update[i]
    return w


def predict(data,w):
    yhat=((data.dot(w[1:])+w[0]) >=0).astype(np.int)
    return yhat

np.random.seed(42)
w=fit(xtrain,ytrain)
w
test=testdf.values
test
xtest=test[:,:-1]
xtest
ytest=test[:,-1]
ytest.shape
yhat=predict(xtrain,w)
yhat
(ytrain==yhat).sum()

yhat=predict(xtest,w)
yhat#passing the test value
(ytest==yhat).sum()


###1 gradient decent
###2 stocastic gradient decent (randomly chosing one mismatched record for chnaging rather then full mismatched)
###mini batch decent

##############################################################


sns.lmplot(x="length",y="width",hue="bugtype",fit_reg=False,data=testdf)







w=fit(xtrain,ytrain)   
   






fig=plt.figure(figsize=(6,5))
plt.plot(df[df.bugtype==0]["length"],df[df.bugtype==0]["width"],"b^")
plt.plot(df[df.bugtype==1]["length"],df[df.bugtype==1]["width"],"ro")

testdf=pd.DataFrame(testdata,columns=["length","width","bugtype"])

fig=plt.figure(figsize=(6,5))
plt.plot(testdf["length"],testdf["width"],"kD")
