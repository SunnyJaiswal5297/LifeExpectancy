# -*- coding: utf-8 -*-
"""
Created on Sat May  9 10:30:20 2020

@author: sunny jaiswal
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

"""creating a data frame"""
df = pd.read_csv('Life Expectancy Data.csv')

"""EDA"""
"""Dimensions of dataframe """
print(df.shape)

"""To print list of all attributes/coulmns"""
print(df.columns)

"""To present the info about dataframe"""
print(df.info)

"""counting the no. of null values"""
print(df.isnull().sum())

"""Repalcing null values with median"""
df.fillna(df.median(),inplace = True)
print(df.isnull().sum())

"""deleting the non numeric values"""
df = df.drop(['Country','Year','Status'], axis=1)

# labels(y) and data(X_all)
y = df['Life expectancy '].values
X_all = df.drop(['Life expectancy '], axis=1).values

# splitting the data to train and test parts
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_all, y, test_size=0.3, random_state=42)

# create the model
from sklearn.linear_model import LinearRegression
ln_reg_all = LinearRegression()

# fitting the model to the train data
ln_reg_all.fit(X_train, y_train)

# predicting the data
y_pred = ln_reg_all.predict(X_test)

"""Calculating R-square value"""
def r_square(y_pred,y_test,y):
	y_mean = np.mean(y)
	print('Life expectancy mean value:',y_mean)
	t1=t2=0
	for i in y_pred:
		t1 += ((i-y_mean)**2)
	print(t1)
	for j in y_test:
		t2 += ((j-y_mean)**2)
	print(t2)
	return t1/t2

"""Model Evaluation"""
print('R-squared value ',r_square(y_pred,y_test,y))
from sklearn.metrics import mean_absolute_error
print('MAE : ',mean_absolute_error(y_test,y_pred))
print('RMSE : ',np.sqrt(np.mean((y_pred-y_test)**2)))