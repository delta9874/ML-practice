#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 21 11:44:03 2019

@author: delta
"""

Clustering:- unsupervised learning 
clustering === group, there is no y to predict ,
               (is there any grouping ,how may ?,based on what?)
               
clustering :a> K-means b>hierarchical
k means :
    based on distances (euciledian )

import pandas as pd
import numpy as np
from sklearn import preprocessing
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import metrics

from sklearn import model_selection


from sklearn.model_selection import GridSearchCV
from sklearn import cluster

df=pd.read_csv("/home/delta/dataset/wcd_cluster.csv")
df.sample(10)
df.info()

wcd=df.iloc[:,2:]
wcd.info()

model=cluster.KMeans(n_clusters=4)
model.fit(wcd)
type(model.labels_)#numpy.ndarray
np.unique(model.labels_)#array([0, 1, 2, 3], dtype=int32) ,labels fro different groups
(model.labels_==0).sum()













