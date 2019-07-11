#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  8 11:51:39 2019

@author: delta
"""

import numpy as np
from sklearn import linear_model
np.random.seed(42)
m=1000
X=2*np.random.rand(m,1)
y=4+3*X+np.random.rand(m,1)


X_b=np.c_[np.ones((m,1)),X]
X_b[:5]
model=linear_model.LinearRegression()
model.fit(X_b,y)
print("#########scikit regression##############")
print(model.intercept_)
print(model.coef_[0])



n_iter=10000
lr=0.001 
 

theta=np.random.rand(2,1)
theta


for i in range(n_iter):
    gradients=(2/m)*X_b.T.dot(X_b.dot(theta)-y)
    theta=theta-lr*gradients

print("###########gradient######")
print(theta)      


np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T.dot(y.reshape(-1,1)))































#linear regression normal equation
#normal equation for reg regression 


























