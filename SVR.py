# -*- coding: utf-8 -*-
"""
Created on Sun Aug 30 08:56:46 2026

@author: Barış
"""

# Support Vector Regression

# Scaler'a ihtiyaç duyan bir yöntem. Nedeni outlier'a çok duyarlı.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm

maaslar = pd.read_csv("data/maaslar.csv")

x = maaslar.iloc[:, 1:2].values
y = maaslar.iloc[:, 2:].values

sc1 = StandardScaler()
x_olcekli = sc1.fit_transform(x)
sc2 = StandardScaler()
y_olcekli = sc2.fit_transform(y)

svr_reg = SVR(kernel="rbf")
svr_reg.fit(x_olcekli, y_olcekli.ravel())

plt.scatter(x_olcekli, y_olcekli)
plt.plot(x_olcekli, svr_reg.predict(x_olcekli), color="red")
plt.show()

print(svr_reg.predict(np.array(11).reshape(1, -1)))
print(svr_reg.predict(np.array(6.6).reshape(1, -1)))

