#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  8 15:32:29 2019

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
from sklearn import datasets
import matplotlib

data=datasets.load_digits()
images=data["images"]
plt.imshow(images[16],cmap=matplotlib.cm.binary,interpolation="nearest")
y=data["target"]
y[0]
y[5]
y[16]
len(y)
X=data["data"]
X[0]
X[0].shape# (64,)

y_tr=(y==5).astype(np.int)

df=pd.DataFrame(X,y_tr)

df.info()
y_tr

(y_tr==1).sum()/y_tr.shape[0]




Xtrain,Xtest,ytrain,ytest=model_selection.train_test_split(X,y_tr,test_size=0.25,random_state=42)




(ytrain==1).sum()/ytrain.shape[0]
(ytest==1).sum()/ytrain.shape[0]

model=linear_model.LogisticRegression()
model.fit(Xtrain,ytrain)
Xtrain.shape
predict=model.predict(Xtest)

metrics.accuracy_score(ytest,predict)#0.98888888888888893
metrics.precision_score(ytest,predict)# 0.94999999999999996
metrics.recall_score(ytest,predict)#0.96610169491525422















