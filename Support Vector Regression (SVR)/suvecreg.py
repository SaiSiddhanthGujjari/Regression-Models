# -*- coding: utf-8 -*-
"""
Created on Wed Apr 18 22:08:42 2018

@author: Sai Siddhanth
"""

#Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')

#Creating matrix of features and an dependent vector
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:,2].values

#Splitting the dataset into training set and test set
"""from sklearn.cross_validation import train_test_split
train_X,test_X,train_y,test_y = train_test_split(X,y , test_size = 0.2, random_state = 0)"""


#Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
sc_y = StandardScaler()
X = sc_X.fit_transform(X)
y = sc_y.fit_transform(y)

#Fitting SVR to the dataset
from sklearn.svm import SVR
regressor = SVR(kernel = 'rbf')
regressor.fit(X,y)



#Predicting a new result with polynomial regression
regressor.predict(6.5)

#Visualzing the SVR results
plt.scatter(X, y, color = 'red')
plt.plot(X, regressor.predict((X)), color = 'blue')
plt.title('Level vs Salary (Regression)')
plt.xlabel('Level')
plt.ylabel('Salary')
plt.show()

