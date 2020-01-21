# -*- coding: utf-8 -*-
"""
Created on Fri Apr  6 14:02:20 2018

@author: Sai Siddhanth
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('Position_Salaries.csv')

X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:,2].values

#Linear Regression
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X, y)

#Polynomial Regression
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 4)
X_poly = poly_reg.fit_transform(X)
lin_reg2 = LinearRegression()
lin_reg2.fit(X_poly,y)

#Visualizing Linear Regression
plt.scatter(X, y, color = 'red')
plt.plot(X, lin_reg.predict(X), color = 'blue')
plt.title('Level vs Salary (Linear Regression)')
plt.xlabel('Level')
plt.ylabel('Salary')
plt.show()

#Visualzing Polynomial Regression
X_grid = np.arange(min(X),max(X),0.1)
X_grid= X_grid.reshape(len(X_grid),1)
plt.scatter(X, y, color = 'red')
plt.plot(X_grid, lin_reg2.predict(poly_reg.fit_transform(X_grid)), color = 'blue')
plt.title('Level vs Salary (Polynomial Regression)')
plt.xlabel('Level')
plt.ylabel('Salary')
plt.show()

#Predicting a new result with linear regression
lin_reg.predict(6.5)

#Predicting a new result with polynomial regression
lin_reg2.predict(poly_reg.fit_transform(6.5))
