#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 25 14:29:56 2019

@author: delta
"""
import numpy as np
print(np.__version__)
arr1=np.array([2,2323,32,23,322,232],dtype=np.int)
arr1=arr1.reshape(2,3)
arr1


list=[11,343,35,46,5,57,76,8]
arr1=np.array(list,dtype=np.float)# makes the array from the list#dtpye is specified if we want to specify the data type or numpy decide own its own
type(arr1) 
 
arr1.dtype#gives the data type of element
print(arr1)
arr1[0]
arr1.shape  #
arr1.shape[0] #gives the size of the array
#another method to make array
arr2=np.arange(-1,1,.05)
print(arr2)
arr2.shape[0]
arr2[0:10]  #array slicing
arr2[-10:]


arr2=np.arange(6,24,2,dtype=float)
arr2=arr2.argsort()
arr=[122,3,13,213,13,131]
p=arr2.iteamsize
print (p)

type(arr)
arr2=np.dtype(arr)
arr2
arr2.reshape(4,6)#reshape in matrix form

arr2=np.arange(24).reshape(4,6)

arr3=np.zeros((4,4))
arr4=np.ones((4,4))
arr5=np.eye(4)#identity matrix
arr6=np.random.random((4,6))#ramdon element array

arr7=np.random.randn(4,6)#its is giving those random value from a set of no whose mean is 0,and sd 1,normally distributed
arr8=np.random.normal(4)#we can define mean and sd and get the array
arr8


np.random.seed(10)#gives the same random no
arr6=np.random.random(4)
arr6
arr6=np.arange(10)
arr=arr6%10 #broadcasting
arr4=arr6+4
arr4.astype(np.float)
arr4
arr4.astype(np.str)
np.append(arr4,new newarr)




import numpy as np
arr1=np.arange(1,10,.5,dytpe=float)
arr1=arr1*2
arr1
arr1=np.linspace(1,10,20,retstep=True,dtype=float)
arr1=arr1*2

arr1=np.logspace(1,10,20,base=2)#base^start to base^stop
arr1

a=np.arange(10)
a
q=a[slice(2,8,2)]
q
arr=np.arange(24).reshape(8,3)
arr[2:5]
arr[3:]
arr[...,0]
arr1=np.arange(1,26).reshape(5,5)
arr1
np.random.shuffle(arr1)#inplace shuffle
arr1

arr1[1][2]
arr1[1,2]
arr2=arr1[1:3,2:4]
arr2
arr1[1:3,2:4]=100#broadcasting in the slice created
arr1

(0,2)(1,1)(2,2)<=99
arr1[[0,1,2],[2,1,2]]=99#arr[[rows],[coloumn]]

cc=np.array([10,11,10,12,11,10])
arr1=np.zeros([6,3])
arr1
k=np.unique(cc).shape[0]
arr1=np.zeros([cc.shape[0],k])#one hot enccoded matrix
arr1[[np.arange(cc.shape[0])],[cc%10]]=1

arr2=np.arange(5)
arr2
np.random.shuffle(arr2)
arr2
arr2>2#arr2[0]>2,arr2[1]>2......
boolarr=np.array([True,False,True,False,False])
arr2
arr2[boolarr]#where the boolean value is true those cells values are return
#boolean indexng

arr2[arr2>2]
arr1=np.array([12,-10,3])
arr2=np.array([-12,20,40])
arr3=arr1.dot(arr2)

arr3

arr1=np.arange(6).reshape(3,2)
arr1
arr2=np.arange(2).reshape(2,1)
arr1.dot(arr2)


#numpy global function

sqrtarr4=np.sqrt(arr3)
sqrtarr4
arr4=np.random.random(5)
arr4=arr4*100000
arr4
logarr4=np.log(arr4)
logarr4

##########################w1x1 + w2x2 +w3
w=np.random.random(3)
w 
x=np.random.random((300,2)) *2 -1
x
sign=np.random.random(300)
sign
sign=x.dot(w[:2])+w[2]

result=np.sign(x.dot(w[:2])+w[2])
result
result[result==-1].shape

################################3
w[a,b, c]
  0 1 2
























































