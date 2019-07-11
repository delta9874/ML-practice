#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 12 11:33:58 2019

@author: delta
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import tree
from sklearn import utils
from sklearn import model_selection

df=pd.read_csv("/home/delta/dataset/iris.csv")
df=utils.shuffle(df)
df.info()
X=df.drop("class",axis=1)
y=df["class"]
y.replace({"Iris-setosa":1,"Iris-versicolor":2,"Iris-virginica":3},inplace=True)


y.value_counts()

Xtrain,Xtest,ytrain,ytest=model_selection.train_test_split(X,y,test_size=0.15,random_state=42)  
trmodel=tree.DecisionTreeClassifier(criterion="entropy")
trmodel.fit(Xtrain,ytrain)
tree.export_graphviz(trmodel,out_file="/home/delta/dataset/tree_ir.dot",feature_names=Xtrain.columns.values,class_names=["setosa","versicolor","virginica"])

trpredict=trmodel.predict(Xtest)










