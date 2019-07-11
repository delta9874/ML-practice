#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 11 14:51:56 2019

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



df=pd.read_csv("/home/delta/dataset/pima_diabetics.csv")
df.info()#768 non-null int64
df["class"].value_counts()
df.head()

df.isnull().sum()
fig,axes=plt.subplots(nrows=4,ncols=3,figsize=(6,6))
axes1=axes.flatten()
index=0
for col in df.columns.values:
    sns.boxplot(y=col,data=df,ax=axes1[index])
    index+=1
plt.tight_layout()


df[df['bp']==0].shape
df.columns().sum()
######checking zero
for i in df.columns:
    print(i)
    print(df[df[i]==0].shape)
 
#droping 0 in bp and bmi
df.drop(df["bp"]==0,axis=0,inplace=True)    
    
for index,record in df.iterrows():
    if record["bp"]==0:
        df.drop(index,axis=0,inplace=True)    
      
for index,record in df.iterrows():
    if record["bmi"]==0:
        df.drop(index,axis=0,inplace=True)       
    
df.info() #727 non-null int64  
df.describe()



####
plt.figure(figsize=(8,8))
sns.heatmap(cbar=False,annot=True,data=df.corr()*100)
plt.title('% Corelation Matrix')
plt.show()

######
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
########

#####applying model sgd and lr on raw data
X=df.drop("class",axis=1) 
X.info()
y=df["class"]
y.info()



Xtest,Xtrain,ytest,ytrain=model_selection.train_test_split(X,y,test_size=0.15,random_state=42)  



###lr
start=datetime.now()    
lrmodel=linear_model.LogisticRegression()
lrmodel.fit(Xtrain,ytrain)
end=datetime.now()
print(end -start)
lrpredicted=lrmodel.predict(Xtest)
print(metrics.confusion_matrix(ytest,lrpredicted))
lrpredicted_probs=lrmodel.predict_proba(Xtest)    

metrics.accuracy_score(ytest,lrpredicted)#0.726094003241 

printresult(ytest,lrpredicted)

distances=lrmodel.decision_function(Xtest)
precission,recall,thresh=metrics.precision_recall_curve(ytest,distances) 
plt.plot(thresh,precission[:-1],color="b")
plt.plot(thresh,recall[:-1],color="g") 
plt.plot(precission,recall)

# =============================================================================
# [[353  56]
#  [113  95]]
# accuracy : 0.7261
# precision : 0.6291
# recall : 0.4567
# f1-score : 0.5292
# AUC : 0.6599
# =============================================================================




 
### sgd
start=datetime.now()
print(start)
sgdmodel=linear_model.SGDClassifier(random_state=42)
sgdmodel.fit(Xtrain,ytrain)
end=datetime.now()
print(end -start)
sgdpredicted=sgdmodel.predict(Xtest)
print(metrics.confusion_matrix(ytest,sgdpredicted))
printresult(ytest,sgdpredicted) 
distances=sgdmodel.decision_function(Xtest)
precission,recall,thresh=metrics.precision_recall_curve(ytest,distances) 
plt.plot(thresh,precission[:-1],color="b")
plt.plot(thresh,recall[:-1],color="g") 
plt.plot(recall,precission)
# =============================================================================
# [[199 210]
#  [ 59 149]]
# accuracy : 0.5640
# precision : 0.4150
# recall : 0.7163
# f1-score : 0.5256
# AUC : 0.6014    
# =============================================================================


###predicting insulin
df1=df.copy()
df2=df[["pgc","tsft","bmi","dpf","insulin"]]



for index,record in df1.iterrows():
    if record["insulin"]==0:
        df.drop(index,axis=0,inplace=True) 







###############################
#selecting feature
sns.heatmap(df[df.columns[:-1]].corr(),annot=True)
fig=plt.gcf()
fig.set_size_inches(10,8)
plt.show()       


X=df[["pgc","np","age","dpf","bmi",'class']]
X.info()
df.info()

for i in X.columns:
    print(i)
    print(df[df[i]==0].shape)
Y.value_counts()

##standerdising

X.describe()
X=StandardScaler().fit_transform(X)

features=X[X.columns[:5]]
features_standard=StandardScaler().fit_transform(features)# Gaussian Standardisation
X=pd.DataFrame(features_standard,columns=[['pgc','np','age','dpf','bmi']])
features=X[X.columns]
features.describe()
Y.info()
X['class'].value_counts()
Y=df['class']
X.drop("class",axis=1,inplace=True)


#####lr model
Xtrain,Xtest,ytrain,ytest=model_selection.train_test_split(X,Y,test_size=0.15,random_state=42)
Xtrain.info()
ytrain.value_counts()
ytest.value_counts()
lrmodel=linear_model.LogisticRegression()
lrmodel.fit(Xtrain,ytrain)
lrpredicted=lrmodel.predict(Xtest)
printresult(ytest,lrpredicted)
distances=lrmodel.decision_function(Xtest)
precission,recall,thresh=metrics.precision_recall_curve(ytest,distances) 
plt.plot(thresh,precission[:-1],color="b")
plt.plot(thresh,recall[:-1],color="g") 
plt.plot(precission,recall)
# =============================================================================
# accuracy : 0.7727
# precision : 0.7179
# recall : 0.6667
# f1-score : 0.6914
# AUC : 0.7525
#     
# =============================================================================
    
    
###sgd
sgdmodel=linear_model.SGDClassifier(random_state=42)
sgdmodel.fit(Xtrain,ytrain)
sgdpredicted=sgdmodel.predict(Xtest)
printresult(ytest,sgdpredicted)
distances=sgdmodel.decision_function(Xtest)
precission,recall,thresh=metrics.precision_recall_curve(ytest,distances) 
plt.plot(thresh,precission[:-1],color="b")
plt.plot(thresh,recall[:-1],color="g") 
plt.plot(precission,recall)   
# =============================================================================
# accuracy : 0.7727
# precision : 0.7297
# recall : 0.6429
# f1-score : 0.6835
# AUC : 0.7479
# =============================================================================


######decision tree
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




df=pd.read_csv("/home/delta/dataset/pima_diabetics.csv")

X=df.drop("class",axis=1)
y=df["class"]


Xtrain,Xtest,ytrain,ytest=model_selection.train_test_split(X,y,test_size=0.15,random_state=42)
trmodel=tree.DecisionTreeClassifier(max_depth=3,min_samples_split=30)
lrmodel=linear_model.LogisticRegression()
results=model_selection.cross_val_score(trmodel,X,y,cv=5,scoring="recall")
results
#array([ 0.48148148,  0.68518519,  0.57407407,  0.67924528,  0.52830189])
#leave one out cross validiation

resultlr=model_selection.cross_val_score(lrmodel,X,y,cv=5,scoring="recall")
resultlr
#array([ 0.53703704,  0.61111111,  0.44444444,  0.60377358,  0.50943396])

results.mean()# 0.5896575821104123
results.std()#0.08106900137688107
resultlr.mean()#0.54116002795248086
resultlr.std()# 0.061953301672862747

######################grid search :hype parameter tuning
depth=list(range(3,11))
depth# [3, 4, 5, 6, 7, 8, 9, 10]
grid={"max_depth":depth}
model=tree.DecisionTreeClassifier()
gmodel=model_selection.GridSearchCV(model,grid,scoring="recall")
gmodel.fit(X,y)
print(gmodel.best_params_)#{'max_depth': 7
print(gmodel.best_score_)
bestmodel=gmodel.best_estimator_
bestmodel.fit(Xtrain,ytrain)
predict=bestmodel.predict(Xtest)

printresult(ytest,predict)
# =============================================================================
# [[61 15]
#  [13 27]]
# accuracy : 0.7586
# precision : 0.6429
# recall : 0.6750
# f1-score : 0.6585
# AUC : 0.7388
# 
# =============================================================================
















