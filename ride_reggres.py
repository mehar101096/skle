# -*- coding: utf-8 -*-
"""
Created on Fri May 17 18:45:17 2019

@author: DELL
"""

import numpy as np
import matplotlib.pyplot as plt 

import pandas as pd  
import seaborn as sns 


from sklearn.datasets import load_boston
boston_dataset=load_boston()
boston=(pd.DataFrame(boston_dataset.data,columns=boston_dataset.feature_names))

boston['MEDV']=boston_dataset.target


df=boston

x=df[['LSTAT','RM']]
y=df['MEDV']


plt.scatter(df['RM'],df['MEDV'])


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.21,random_state=2)

from sklearn.linear_model import LinearRegression
clf = LinearRegression()
clf.fit(x_train,y_train)
clf.predict(x_test)
clf.score(x_test,y_test)