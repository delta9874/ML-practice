#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 11 11:53:57 2019

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

df=pd.read_csv("/home/delta/dataset/mnist_train.csv",header=None)
df.shape
df1=pd.read_csv("/home/delta/dataset/mnist_test.csv",header=None)
df1.shape
data=df.values ###numpy  arry

plt.imshow(data[0,1:].reshape(28,28))


#####shuffle
np.random.seed(42)
np.random.shuffle(data)
######
Xtrain=data[:,1:]
ytrain=data[:,0]
ytrain
ytrain=(ytrain==5).astype(np.int)
ytrain[:10]
data1=df1.values

np.random.seed(42)
np.random.shuffle(data1)
######
Xtest=data1[:,1:]
ytest=data1[:,0]

ytest=(ytest==5).astype(np.int)
ytest[:10]
ytest.shape

start=datetime.now()
print(start)
##########lr model
lrmodel=linear_model.LogisticRegression()
lrmodel.fit(Xtrain,ytrain)
end=datetime.now()
print(end -start)#0:04:30.148196
lrpredicted=lrmodel.predict(Xtest)
print(metrics.confusion_matrix(ytest,lrpredicted))
lrpredicted_probs=lrmodel.predict_proba(Xtest)

#######sgd
start=datetime.now()
print(start)
sgdmodel=linear_model.SGDClassifier(random_state=42)
sgdmodel.fit(Xtrain,ytrain)
end=datetime.now()
print(end -start)#0:00:01.568268
sgdpredicted=sgdmodel.predict(Xtest)
print(metrics.confusion_matrix(ytest,sgdpredicted))
sgdpredicted_probs=sgdmodel.predict_proba(Xtest)#probability estimates are not available for loss='hinge'



distances=sgdmodel.decision_function(Xtest)

sgdpred1=(distances>=0).astype(np.int)
print(metrics.confusion_matrix(ytest,sgdpred1))
print(metrics.classification_report(ytest,sgdpred1))
print(metrics.precision_score(ytest,sgdpred1))
print(metrics.recall_score(ytest,sgdpred1))



distances.max()

sgdpred1=(distances>=100000).astype(np.int)
print(metrics.confusion_matrix(ytest,sgdpred1))
print(metrics.classification_report(ytest,sgdpred1))
print(metrics.precision_score(ytest,sgdpred1))
print(metrics.recall_score(ytest,sgdpred1))



sgdpred1=(distances>=-100000).astype(np.int)
print(metrics.confusion_matrix(ytest,sgdpred1))
print(metrics.classification_report(ytest,sgdpred1))
print(metrics.precision_score(ytest,sgdpred1))
print(metrics.recall_score(ytest,sgdpred1))

precision,recall,thresh=metrics.precision_recall_curve(ytest,distances)
plt.plot(thresh,precision[:-1],color="g")
plt.plot(thresh,recall[:-1],color="b")

plt.plot(recall,precision)



###TPR==true positive rate == recall==TP/(FN+TP)
###FPR==False positie rate ==  FP/(FP+TN)
#tpr is directly prop to fpr
#AUC score must 
metrics.roc_auc_score(ytest,sgdpredicted)
fpr,tpr,thres=metrics.roc_curve(ytest,sgdpredicted)
##Roc curve
plt.plot([0,1],[0,1],color="k")
plt.plot(fpr,tpr,color="b")










####loss="log"
start=datetime.now()
print(start)
sgdmodel=linear_model.SGDClassifier(loss="log",random_state=42)
sgdmodel.fit(Xtrain,ytrain)
end=datetime.now()
print(end -start)#0:00:01.568268
sgdpredicted=sgdmodel.predict(Xtest)
print(metrics.confusion_matrix(ytest,sgdpredicted))
sgdpredicted_probs=sgdmodel.predict_proba(Xtest)













