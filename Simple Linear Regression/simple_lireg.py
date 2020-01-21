# -*- coding: utf-8 -*-
"""
Created on Fri Mar 23 09:31:34 2018

@author: Sai Siddhanth
"""

# -*- coding: utf-8 -*-
"""
Created on Sun Mar 11 14:07:18 2018

@author: Sai Siddhanth
"""

#Importing libraries
#numpy used in making mathematical models
import numpy as np
# matplotlib.pyplot is used in graphs and plotting 
import matplotlib.pyplot as plt
#pandas is used for importing datasets
import pandas as pd


#importing the dataset

dataset = pd.read_csv('Salary_Data.csv')

#creating matrix of features....independent variables
X = dataset.iloc[:, :-1].values

#creating matrix of dependent variables
y = dataset.iloc[:,1].values

#Splitting the dataset into training set and test set
from sklearn.cross_validation import train_test_split
train_X,test_X,train_y,test_y = train_test_split(X,y , test_size = 1/3, random_state = 0)

"""#Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
train_X = sc_X.fit_transform(train_X)
test_X = sc_X.fit_transform(test_X)"""


#Fitting simple linear regression model to our training set ....creating a machine(regressor) that will learn from train observations 
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(train_X, train_y)

#Predicting the Test Set Results
pred_y = regressor.predict(test_X)

#Visualizing the training set Results
plt.scatter(train_X, train_y, color = 'red')
plt.plot(train_X, regressor.predict(train_X), color = 'blue')
plt.title('Salary v Experience(Training Set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()


#Visualizing the test set Results
plt.scatter(test_X, test_y, color = 'red')
#unique simple linear equation learnt using the regressor so the regression line will be the same 
plt.plot(train_X, regressor.predict(train_X), color = 'blue')
plt.title('Salary v Experience(Test Set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()


#Experimenting
plt.scatter(test_X, test_y, color = 'red')
plt.plot(test_X, pred_y , color = 'blue')
plt.title('Salary v Experience(Test Set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()




