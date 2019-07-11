#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 31 11:34:18 2019

@author: delta
"""

    
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt #ploting and visualisation
import seaborn as sns#ploting    
df=pd.read_csv("/home/delta/dataset/StudentsPerformance.csv")
df.info()
df.head()
coldict={"race/ethnicity":"race","parental level of education":"p_edu","test preparation course":"pcourse","math score":"math","reading score":"reading","writing score":"writing"}
df.rename(columns=coldict,inplace=True)
df.info()
df.race.unique()
df.race.value_counts()
df.race=df.race.str.replace("group\s*","")
df.lunch.value_counts()
df.lunch=df.lunch.str.replace("free/reduced","free")
df.p_edu.value_counts()
df.race.value_counts()
# =============================================================================
li=df.race.unique()
li
li.sort()
li
z="1"
for i in li:
    df.race=df.race.str.replace(str(i),str(z))
    int(z)+=1 
# =============================================================================
    
    
repldict={"A":1,"B":2,"C":3,"D":4,"E":5}#change the groupdata in to numberrs to easy calculation

df.race.replace(repldict,inplace=True)

df.p_edu=df.p_edu.str.upper()
df.p_edu=df.p_edu.str.replace(" ","_")
df.math.describe()
df.math.value_counts()
df[df.math==df.math.min()].shape[0]
df[df.math==df.math.min()].index.values#gives the index of the student who have scored the min value
df.iloc[df[df.math==df.math.min()].index.values,:]#gives the full data



IQR=math.quantile(.75)-math.quantile(.25)
75% percentile + 1.5*iqr#theoritical maximum value
#values greater then this is outlairs extremly large val ,doesnt go with the set trend
25% percentile -1.5*iqr #theoritical minimun value,outliars

m=df.math
iqr=m.quantile(.75)-m.quantile(.25)
k=m.quantile(.75)+1.5*iqr
k
l=m.quantile(.25)-1.5*iqr
l
m[(m>k)]
m[(m<l)]
m[(m>m.quantile(.75)+1.5*iqr)|(m<m.quantile(.25)-1.5*iqr)].shape#outliars
outindex=m[(m<l)].index.values
outindex
df.iloc[outindex]

0 25 =ba
26 75=a
76 100=g

def grade(marks):
    if marks>=76:
        return "g"
    elif marks>=26:
        return "a"
    else:
        return "ba"
df["mathgade"]=df.math.map(grade)
df.head()    
###does pcourse help in improving maths
df.math[df.pcourse=="none"].mean()
df.math[df.pcourse=="completed"].mean()    
df.math[df.pcourse=="none"].describe()   
df.math[df.pcourse=="completed"].describe()    
    
#f distribution ,chi statistics,p value,
#########################################visualisation
df.info()
plt.scatter(df.math,df.reading)
plt.xlabel("Math Score")
plt.ylabel("Reading Score")
df.pcourse.replace({"none":0,"completed":1},inplace=True)
plt.scatter(df.math,df.reading,c=df.pcourse)
plt.xlabel("Math Score")
plt.ylabel("Reading Score")
plt.legend()
plt.scatter(df.math,df.reading,s=500,c=df.pcourse)#s is for size
plt.xlabel("Math Score")
plt.ylabel("Reading Score")
plt.legend()    
 
sns.lmplot(x="math",y="reading",data=df,hue="pcourse",fit_reg=False)
sns.lmplot(x="math",y="reading",data=df,hue="pcourse")   #gives regression line
sns.lmplot(x="writing",y="reading",data=df,hue="pcourse",fit_reg=False)
sns.lmplot(x="math",y="writing",data=df,hue="pcourse",fit_reg=False)
sns.lmplot()   
sns.lmplot(x="math",y="reading",data=df,fit_reg=False)   
sns.lmplot(x="math",y="writing",data=df,fit_reg=False)    
df.math.hist()  
df.math.hist(bins=30)
sns.distplot(df.math,bins=30,kde=False)    
df.math.mean()    
####before going for any test check for normality
sns.distplot(df.math,bins=30,hist=False)  
sns.distplot(np.log(df.math),bins=30,hist=False)#various function could be aplied to the data to get to normalize
sns.boxplot(y=df.reading)  #boxplot
sns.boxplot() 
df.p_edu.unique()
sns.boxplot(x="p_edu",y="math",data=df)
sns.boxplot(x="race",y="math",data=df)
plt.xticks(rotation=90)    
    
    
sns.countplot(x="race",data=df)
ax=sns.countplot(x="race",hue="lunch",data=df)    
type(ax)    
ax    
sns.violinplot(y=df.math)    
sns.violinplot(y=df.math,inner="quart")    
sns.violinplot(y="math",x="lunch",inner="quart",data=df)        
sns.violinplot(y="math",x="gender",inner="quart",data=df)
sns.violinplot(y="math",x="lunch",inner="quart",hue="gender",split=True,data=df)













    