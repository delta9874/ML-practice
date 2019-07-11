#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  4 19:05:25 2019

@author: delta
"""


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import linear_model
from sklearn import model_selection
from sklearn import metrics


df1=pd.read_csv("/home/delta/dataset/housing.csv")
df1.shape[0]
df1.info()



fig,axes=plt.subplots(nrows=4,ncols=3,figsize=(6,6))
axes1=axes.flatten()
index=0
for col in df.columns.values:
    sns.boxplot(y=col,data=df,ax=axes1[index])
    index+=1
plt.tight_layout()

############
df = df.rename(columns={'oldName1': 'newName1', 'oldName2': 'newName2'})
# Or rename the existing DataFrame (rather than creating a copy) 
df.rename(columns={'oldName1': 'newName1', 'oldName2': 'newName2'}, inplace=True)
######################
df=df1.rename(columns={"longitude":"long","latitude":"lat","housing_median_age":"hma",
                       "total_rooms":"tr","total_bedrooms":"tb","median_income":"mi",
                       "median_house_value":"mhv","ocean_proximity":"op"})

df.tr.value_counts()
repldict={"<1H OCEAN":1,"INLAND":2,"NEAR OCEAN":3,"NEAR BAY":4,"ISLAND":5}
df.op.replace(repldict,inplace=True)

###############

def iqr(series):#fuction is accepting a series
    return (series.describe()["75%"]-series.describe()["25%"])
    #return series.quantile(q=.75)-series.quantile(q=.25)
df[df.tr<1700].shape[0]



m=df.mhv
m
iqr=m.quantile(.75)-m.quantile(.25)
iqr
m[m>(m.quantile(.75)+1.5*iqr)|(m<(m.quantile(.25)-1.5*iqr)].shape[0]
m[(m>m.quantile(.75)+1.5*iqr)|(m<m.quantile(.25)-1.5*iqr)].shape
########################


sns.heatmap(df.corr(),annot=True)
sns.distplot(df["tb"])


fig,axes=plt.subplots(nrows=4,ncols=3,figsize=(6,6))
axes1=axes.flatten()
index=0
for col in df.columns.values:
    sns.violinplot(y=col,data=df,ax=axes1[index])
    index+=1
plt.tight_layout()
df.info()
df.tb.value_counts()
df.tb.describe()
df.tb.fillna(value=df.tb.mean(),inplace=True)


import math

for col in df.columns.values :
    sns.distplot(df[col])

from sklearn import preprocessing
X_scaled = preprocessing.scale(df.hma)
sns.distplot(X_scaled)
sns.distplot(df.hma)
(X_scaled - df.tb)
sns.distplot(1/(1-np.exp(np.log((df.hma + (df.hma.max() - df.hma.min())/df.hma.mean())))))
sns.distplot(1/(1-np.exp(np.log((df.hma**.45 + (df.hma.max() + df.hma.min())/df.hma.mean())**4 -df.hma.mean()))))



##########################################################


#kaggel
######################################################################

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import linear_model
from sklearn import model_selection
from sklearn import metrics

df=pd.read_csv("/home/delta/dataset/housing.csv")
df.info()
df.head()

df.rename(columns={"longitude":"long","latitude":"lat","housing_median_age":"hma",
                       "total_rooms":"tr","total_bedrooms":"tb","median_income":"mi",
                       "median_house_value":"mhv","ocean_proximity":"op"},inplace=True)

df.info()

print('the number of rows and colums are'+str(df.shape))
[print(i,end='.\t\n') for i in df.columns.values]
df.isnull().sum()#checking the null values
sns.heatmap(df.isnull(),cmap='viridis',cbar=False,yticklabels=False)
df.tb.describe()

#outlairs in tb as we are working on outlairs
m=df.tb
m
m.info()
iqr=m.quantile(.75)-m.quantile(.25)
iqr
m[m>(m.quantile(.75)+1.5*iqr)|(m<(m.quantile(.25)-1.5*iqr)].shape[0]
m[(m>m.quantile(.75)+1.5*iqr)|(m<m.quantile(.25)-1.5*iqr)].shape

sns.boxplot(y="tb",data=df)


######plooting graph
fig,axes=plt.subplots(nrows=4,ncols=2,figsize=(10,10))
axes1=axes.flatten()
index=0
for col in df.columns.values:
    sns.boxplot(y=col,data=df,ax=axes1[index])
    index+=1
plt.tight_layout()

sns.distplot(df[df["tb"].notnull()]["tb"],bins=20,color="green")


grp_op=df.groupby("op")
tb_group=grp_op["tb"]
tb_group.describe()


tb_dict=dict(tb_group.median())
tb_dict
for index,record in df.iterrows():
    if record["tb"]=="nan":
        df["tb"][index]=tb_dict[record["op"]]
df.info()

###########filling data with meadian
def calc_categorical_median(x):
  
    unique_colums_ocean_proximity=x['op'].unique()
    for i in unique_colums_ocean_proximity:
        median=x[x['op']==i]['tb'].median()
        x.loc[x['op']==i,'tb'] =  x[x['op']==i]['tb'].fillna(median)
calc_categorical_median(df)

#################
df.isnull().sum()
sns.pairplot(data=df)

sns.distplot(df["mhv"])

plt.scatter(df['population'],df['mhv'],c=df['mhv'])
plt.scatter()


#####out lairs in mhv
df[df['mhv']>450000]['mhv'].value_counts().head()#500001.0    965 no of out lairs

df=df.loc[df['mhv']<500001,:]
df.info()
df=df[df['population']<25000]
plt.figure(figsize=(15,10))
plt.scatter(df['long'],df['lat'],c=df['mhv'],s=df['population']/10,cmap='viridis')
plt.colorbar()

####################corealation matrix
plt.figure(figsize=(8,8))
sns.heatmap(cbar=False,annot=True,data=df.corr()*100)
plt.title('% Corelation Matrix')
plt.show()

cdf["mhv"].values
###############################

sns.countplot(data=df,x="op")
sns.boxplot(data=df,x="op",y="mhv")
##########################droping op with one hot endco
cdf=pd.get_dummies(df,columns=["op"],prefix="op")
cdf.info()
cdf[:1]
cdf['h/p']=cdf['households']/cdf['population']
cdf.info()
###########################
################################
#training and testing
###############################
X=cdf.drop(["lat","long","mhv"],axis=1)
y=cdf["mhv"]
X.info()
Xtrain,Xtest,ytrain,ytest=model_selection.train_test_split(X,y,test_size=0.15,random_state=42)


model=linear_model.LinearRegression()
model.fit(Xtrain,ytrain)

predict_train=model.predict(Xtrain)


model.score(Xtrain,ytrain)#0.62077079730825779

model.score(Xtest,ytest)# 0.62965199941928518


##########################################################################
###############################################################################
###################################################Day 7th feb

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import linear_model
from sklearn import model_selection
from sklearn import metrics
from sklearn import preprocessing

df=pd.read_csv("/home/delta/dataset/housing.csv")
df.info()
df.isnull().sum()/df.shape[0]
#####3ignoring the null value
df.dropna(inplace=True)
df.shape[0]
sns.distplot(df.median_house_value)
df[df.median_income>=500000].shape#(985, 10)
df1=df[df.median_house_value>=500000]
df1.median_house_value.describe()
df1.median_income.describe()
df.median_income.describe()
sns.distplot(df.median_income)
df[df.median_income>=15].shape#(50, 10)

df.rename(columns={"longitude":"long","latitude":"lat","housing_median_age":"hma",
                       "total_rooms":"tr","total_bedrooms":"tb","median_income":"mi",
                       "median_house_value":"mhv","ocean_proximity":"op"},inplace=True)

sns.lmplot(x="mi",y="mhv",data=df,fit_reg=False)

####corelation 
plt.figure(figsize=(8,8))
sns.heatmap(cbar=False,annot=True,data=df.corr()*100)
plt.title('% Corelation Matrix')
plt.show()

df.corr()["mi"].sort_values(ascending=False)
# =============================================================================
# mi            1.000000
# mhv           0.688355
# tr            0.197882
# households    0.013434
# population    0.005087
# tb           -0.007723
# long         -0.015550
# lat          -0.079626
# hma          -0.118278
# =============================================================================

df.drop(df[df.mhv>500000].index,inplace=True)
df.shape#(19475, 10)

df.op.unique()
sns.boxplot(y="mhv",x="op",data=df)
df.mi.hist(bins=30)

#####strata ===> a range is known as strata,

df["income_cat"]=np.ceil(df["mi"]/1.5)
df.income_cat.hist(bins=30)
df.income_cat.value_counts()
df.mi[:10]
df.income_cat[:10]
###########changes the values of array according to the given condition
arr=np.array([1.2,3.4,6.7,5.1,29.2,232,2,33.8])
np.where(arr>=5.0,5,arr)
#arr=np.array([1.2,3.4,6.7,5.1,29.2,232,2,33.8])
#array([ 1.2,  3.4,  5. ,  5. ,  5. ,  5. ,  2. ,  5. ])
#chnaging in te original values
arr=np.where(arr>=5.0,5,arr)
arr
#array([ 1.2,  3.4,  5. ,  5. ,  5. ,  5. ,  2. ,  5. ])
##########

df["income_cat"]=np.where(df.income_cat>=5.0,5,df.income_cat)
df.income_cat.hist(bins=30)
df.drop(["lat","long"],axis=1,inplace=True)
df.describe()
df1=df.drop("op",axis=1)




X=df1.drop("mhv",axis=1)
y=df1["mhv"]

Xtrain,Xtest,ytrain,ytest=model_selection.train_test_split(X,y,test_size=0.25,random_state=42)
Xtrain.mi.hist(bins=30)
Xtest.mi.hist(bins=30)
df.mi.hist(bins=30)
Xtrain.income_cat.hist(bins=30)
Xtest.income_cat.hist(bins=30)
#stratifit sampling in case random data split is different in test and train 
sns.heatmap(cbar=False,annot=True,data=X.corr()*100)


model=linear_model.LinearRegression()
model.fit(Xtrain,ytrain)

predict_train=model.predict(Xtrain)


def report (_Xtrain,_Xtest,_ytrain,_ytest):
    model=linear_model.LinearRegression()
    model.fit(_Xtrain,_ytrain)
    pred_train=model.predict(_Xtrain)
    pred_test=model.predict(_Xtest)
    print("train rmse:",np.sqrt(metrics.mean_squared_error(_ytrain,pred_train)))
    print("train rmse:",np.sqrt(metrics.mean_squared_error(_ytest,pred_test)))
    print("train score:",model.score(_Xtrain,_ytrain))
    print("test score:",model.score(_Xtest,_ytest))
    print("true test std:",np.std(_ytest))
    print("predicted test std:",np.std(pred_test))
    return model


report(Xtrain,Xtest,ytrain,ytest)
# =============================================================================
# train rmse: 68043.4684245
# train rmse: 67383.0891964
# train score: 0.514580340071
# test score: 0.525379967531
# true test std: 97808.78974151555
# predicted test std: 71627.8521657
# =============================================================================

sns.heatmap(cbar=False,annot=True,data=X.corr()*100)
##################new feature 
Xtrain1=Xtrain.copy()
Xtest1=Xtest.copy()
Xtrain["br_per_households"]=Xtrain.tb/Xtrain.households
Xtrain.info()
Xtrain.drop(["tb","households","mi"],axis=1,inplace=True)

Xtest["br_per_households"]=Xtest.tb/Xtest.households
Xtest.info()
Xtest.drop(["tb","households","mi"],axis=1,inplace=True)

report(Xtrain,Xtest,ytrain,ytest)
# =============================================================================
# train rmse: 74874.2813219
# train rmse: 73877.1074576
# train score: 0.412226760698
# test score: 0.429489035063
# true test std: 97808.78974151555
# predicted test std: 62969.1039835
# =============================================================================
Xtrain.info()
Xtrain2=Xtrain.copy()
Xtest2=Xtest.copy()
df1.info()
df1["rmperhouseholds"]=df1.tr/df1.households
df1["bedrmperrm"]=df1.tb/df1.tr
df1["populationperhousehols"]=df1.population/df1.households

X=df1[["hma","income_cat","rmperhouseholds","bedrmperrm","populationperhousehols"]]
y=df1["mhv"]
X.info()

Xtrain,Xtest,ytrain,ytest=model_selection.train_test_split(X,y,test_size=0.25,random_state=42)

report(Xtrain,Xtest,ytrain,ytest)

# =============================================================================
# train rmse: 72877.1888964
# train rmse: 72121.963938
# train score: 0.443163493835
# test score: 0.456274979165
# true test std: 97808.78974151555
# predicted test std: 65884.0337431
# =============================================================================



df1.info()
df2=df1.copy()
df2["rmperhouseholds"]=df1.tr/df1.households
df2["bedrmperrm"]=df1.tb/df1.tr
df2["populationperhousehols"]=df1.population/df1.households

X=df2.drop(["mi","mhv"],axis=1)
y=df2["mhv"]
X.info()

Xtrain,Xtest,ytrain,ytest=model_selection.train_test_split(X,y,test_size=0.25,random_state=42)

model=report(Xtrain,Xtest,ytrain,ytest)
model
# =============================================================================
# train rmse: 70650.1416238
# train rmse: 69990.0114831
# train score: 0.476676124374
# test score: 0.487945296963
# true test std: 97808.78974151555
# predicted test std: 68138.2366217
# =============================================================================

from sklearn import feature_selection
sfm=feature_selection.SelectFromModel(model)
#sfm=feature_selection.SelectFromModel(linear_model.LinearRegression())#pass the model or direct regression could be done here
sfm.fit(Xtrain,ytrain)
Xtrain.columns.values[sfm.get_support()]
# =============================================================================
# array(['income_cat', 'bedrmperrm'], dtype=object)
# =============================================================================
df2.info()

X=df2[["income_cat","bedrmperrm"]]
y=df2["mhv"]
X.info()

Xtrain,Xtest,ytrain,ytest=model_selection.train_test_split(X,y,test_size=0.25,random_state=42)

model=report(Xtrain,Xtest,ytrain,ytest)
# =============================================================================
# train rmse: 74918.9745219
# train rmse: 74348.0850436
# train score: 0.41152485594
# test score: 0.422191662957
# true test std: 97808.78974151555
# predicted test std: 63617.4845349
# 
# =============================================================================

##########################feature selection#f testing ,anova
fval,pval=feature_selection.f_regression(Xtrain,ytrain)
for col,f,p in zip(Xtrain.columns,fval,pval):
    print(col,f,p)



skb=feature_selection.SelectKBest(k=4,score_func=feature_selection.f_regression)
skb.fit(Xtrain,ytrain)
Xtrain.columns.values[skb.get_support()]

Xtrain2=Xtrain[Xtrain.columns.values[skb.get_support()]]
Xtest2=Xtest[Xtest.columns.values[skb.get_support()]]

report(Xtrain2,Xtest2,ytrain,ytest)
# =============================================================================
# train rmse: 74859.0403202
# train rmse: 74301.1740215
# train score: 0.412466024196
# test score: 0.422920586326
# true test std: 97808.78974151555
# predicted test std: 63606.7531305
# =============================================================================

df.info()
df["rmperhouseholds"]=df.tr/df.households
df["bedrmperrm"]=df.tb/df.tr
df["populationperhousehols"]=df.population/df.households

df_enc=pd.get_dummies(df,columns=["op"],prefix="op")
df_enc.info()

X=df_enc.drop(["mi","mhv"],axis=1)
y=df_enc["mhv"]

Xtrain,Xtest,ytrain,ytest=model_selection.train_test_split(X,y,test_size=0.25,random_state=42)
report(Xtrain,Xtest,ytrain,ytest)

# =============================================================================
# 
# train rmse: 64667.3278715
# train rmse: 64162.2938448
# train score: 0.56155582252
# test score: 0.569667651607#model accuracy
# true test std: 97808.78974151555
# predicted test std: 74233.8038799#model variance
# =============================================================================

#std increse the uncertinity


#error=y-(w1x1+w2x2+w3)
 #    =bias^2 +variance +irreducable error

############################################3


















