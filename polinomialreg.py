# -*- coding: utf-8 -*-
"""
Created on Sat Aug 29 09:20:52 2026

@author: Barış
"""

# y = b0 + b1x + b1x^2 + ... + bhx^h + e

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm

maaslar = pd.read_csv("data/maaslar.csv")

#plt.plot(maaslar["Egitim Seviyesi"], maaslar.maas)
#plt.show()

x = maaslar.iloc[:, 1:2]
y = maaslar.iloc[:, 2:]

# Linear Regression
lin_reg = LinearRegression()
lin_reg.fit(x, y)

plt.scatter(x, y)
plt.plot(x, lin_reg.predict(x), color="red")
plt.show()

# Polynomial Regression
poly_reg = PolynomialFeatures(degree=2)
x_poly = poly_reg.fit_transform(x)

lin_reg2 = LinearRegression()
lin_reg2.fit(x_poly, y)

poly_reg2 = PolynomialFeatures(degree=4)
x_poly2 = poly_reg2.fit_transform(x)

lin_reg3 = LinearRegression()
lin_reg3.fit(x_poly2, y)

plt.scatter(x, y)
plt.plot(x, lin_reg2.predict(x_poly), color="red", label="Power 2")
plt.plot(x, lin_reg3.predict(x_poly2), color="orange", label="Power 4")
plt.legend()
plt.show()

print(lin_reg2.predict(poly_reg.fit_transform([[6.6]])))
print(lin_reg2.predict(poly_reg.fit_transform([[11]])))

print(lin_reg3.predict(poly_reg2.fit_transform([[6.6]])))
print(lin_reg3.predict(poly_reg2.fit_transform([[11]])))

