#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 28 15:55:28 2019

@author: delta
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt #ploting and visualisation
import seaborn as sns#ploting

math=pd.Series([85,22,45,42,99],index=["jhon","robert","arya","ned","littlefinger"])
type(math)
math
math["littlefinger"]

for marks in math:
    print (marks)#gives values
matharr=math.values
type(matharr)#gives numpy array from series
matharr
mathindex=math.index
type(mathindex)
mathindex
#to get the values
mathindex=math.index.values
type(mathindex)
math.mean()
math.median()
math.mode()
m=math.quantile(q=.1)
m

IQR=math.quantile(.75)-math.quantile(.25)
75% percentile + 1.5*iqr#theoritical maximum value
#values greater then this is outlairs extremly large val ,doesnt go with the set trend
25% percentile -1.5*iqr #theoritical minimun value,outliars


phy=pd.Series([98,77,np.NaN,90,82],index=["jhon","sam","omega","arya","stark"])
phy1=phy.fillna(value=0)#new series will be crated,used to remove  nan
phy
phy1
phy.fillna(value=0,inplace=True)#to make changes in actual series
phy


med=phy.median()
phy1=phy.fillna(value=med,inplace=True)
phy1
phy.fillna(method="ffill",inplace=True)
phy
phy.fillna(method="bfill",inplace=True)
phy.isnull().sum()#gives the no of nan
phy.isnull().sum()/phy.shape[0]#gives the percentage of nan
phy.sort_values()
phy.sort_values(ascending=False)
phy=pd.Series([98,77,np.NaN,90,82])#,index=["jhon","sam","omega","arya","stark"])
phy.sort_values()


math=pd.Series([85,22,45,42,99],index=["jhon","robert","arya","ned","littlefinger"])
phy=pd.Series([98,77,np.NaN,90,82],index=["jhon","robert","arya","ned","littlefinger"])
d={"math":math,"physics":phy}#seris must share same index
df=pd.DataFrame(d)#collection of different series
df#has two set of index in data frame
data=df.values
type(data)
k=df.columns.values
k
df["math"]#dataframe can be broken down as series
math
df["math"].max()
for col in df.columns.values:
    print("col:",col,"max:",df[col].max())### df[math]  df[phy]


#####day 5##########
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt #ploting and visualisation
import seaborn as sns#ploting

arr1=np.random.random(10)
#astype   for type conversion

phy=pd.Series([98,77,np.NaN,90,82],index=["jhon","sam","omega","arya","stark"])
phy.describe()
phy.describe()["50%"]
phy.value_counts()#gives us the frequency of discrete data
#datafame ===collection of series ,all the series have same index
 

df1=pd.DataFrame((np.random.random((10,4))*100).astype(np.int),columns=["a","b","c","d"])
df1
df1["a"].max()
df1["a","b"]
df1[["a","b"]]#pass it as a list
df1.a#it has some restriction as column name shoould not be python key wordor any space
df1.iloc[0]#index location#if we has provided a index ,so to search for specific index we can type the index
df1.loc[0]#loctaion if no index is provided and index is random generated number
df1.iloc[0,0]#rows ,columns
df1.iloc[:2,:2]#slicing
df1[2:4]#rows slice
df1["abc"]#column name if not present give a error
df1["a"]#checks the column name
df1[df1>10]#boolean indexing ,where it will be less the 10 it is replaced by nan
#creating a extra coloumn
df1["total"]=df1.a+df1.b+df1.c
df1
df2=df1.drop(0,axis=0)#to draop a column(row,axis=0for row,axis 1 for coloumn)used to delete whole column or row
#doesnt change the original data frame#by default axis =0
df1.drop("total",axis=1,inplace=True)
df2=df1#creates a shallow copy ,any inplace changes made in df2 affects the df1
#to avoid memeory wasteage
df2.drop("d",axis=1,inplace=True)
df2=df1.copy()#creats a deep copy


##################################################
def iqr(series):#fuction is accepting a series
    return (series.describe()["75%"]-series.describe()["25%"])
    #return series.quantile(q=.75)-series.quantile(q=.25)
df1.apply( iqr,axis=0)
######if a guys score>150 ===pass else fail
def passfail(series):
     return "pass" if series.sum()>150 else "Fail"
df1.apply(passfail,axis=1) #apply acts on series only,apply accepts a fuction and apply it t a series creted by taking values from either row wise or coloumn wise
    
df1["result"]=df1.apply(passfail,axis=1)
df1
df1["result"].count()
df1.result.value_counts()    
####percentage of fail or pass
df1.result.value_counts()/df1["result"].count()*100
df1.result.value_counts()/df1.shape[0]
###############
def passfail(series):
     return "pass" if series.sum()>150 else "Fail"
df1.apply() #apply acts on series only,apply accepts a fuction and apply it t a series creted by taking values from either row wise or coloumn wise
    
df1["result"]=df1.apply(passfail,axis=1)

df1["trres"]=(df1.result=="pass").astype(int)
df1


#####################file handling
df=pd.read_csv("/home/delta/dataset/bp-age.csv")
df.info()#rapid information about file\
total=df.shape[0]
total
df.describe()
autodf=pd.read_csv("/home/delta/dataset/auto-mpg.data",sep="\\s+",header=None,names=["mpg","cylinders","displacement","hp","weight","acceleration","modelyear","origine","carname"])
autodf.describe()
autodf.info()
#######to display the large file
autodf 
pd.set_option("display.width",1000)
pd.set_option("display.max_columns",50)
pd.set_option("display.max_rows",1000)
autodf

print(pd.set_option.__doc__)#read the documentation
for col in autodf.columns.values:
    print(col,":")
    print(autodf[col].value_counts())
    print("================")
autodf["acceleration"].describe()
autodf.cylinders.unique()#numpy arry of unique value
autodf.mpg.unique()

auto8=autodf[autodf.cylinders==8]
auto8["mpg"].mean()

for i in autodf.cylinders.unique():
    print("cyclinder:",i,autodf[autodf.cylinders==i]["mpg"].mean())
autodf.info()
autodf.modelyear.unique()
for i in autodf.modelyear.unique():
    print("modelyear:",i,autodf[autodf.modelyear==i]["cylinders"].value_counts())



######################visualisation
    
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt #ploting and visualisation
import seaborn as sns#ploting    
df=pd.read_csv("/home/delta/dataset/bp-age.csv")


autodf=pd.read_table("/home/delta/dataset/auto-mpg.data",sep="\\s+",header=None)


df
df["bp"].mean()
df.info()
df.describe()
pd.read_csv.__doc__.split()
df
a=df.bp.unique()
type(a)
a.size
for i in df.bp.unique():
    t+="bp:",i,df[df.bp==i]["weight"].value_counts()

type(t)
sorted(t)
t.shape()
df.shape
movies=pd.read_csv('http://bit.ly/imdbratings')
movies.head()
t.sort_values()

t=df["bp"].sort_values()#series sorting
df.sort_values(["weight","age","bp"])#dataframe sorting
type(t)


￼
￼
￼
￼
￼

￼
￼
































































































