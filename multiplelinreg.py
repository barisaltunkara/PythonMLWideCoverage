# -*- coding: utf-8 -*-
"""
Created on Wed Aug 26 06:14:34 2026

@author: Memocantrolovski
"""

# y = b0 + b1x1 + b2x2 + b3x3 + e

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm

veriler = pd.read_csv("data/veriler.csv")

# Label Encoder
ulke = veriler.iloc[:, 0:1].values.copy()
#print(ulke)

le = preprocessing.LabelEncoder()
ulke[:, 0] = le.fit_transform(veriler.iloc[:, 0])
#print(ulke)

cinsiyet = veriler.iloc[:, 4].values
cinsiyet = le.fit_transform(cinsiyet)
#print(cinsiyet)

# One Hot Encoder
ohe = preprocessing.OneHotEncoder()
ulke = ohe.fit_transform(ulke).toarray()
#print(ulke)

# Veri Birlestirme
sonuc = pd.DataFrame(data=ulke, index=range(22), columns=["fr", "tr", "us"])
#print(sonuc)

sonuc2 = pd.DataFrame(data=veriler.iloc[:, 1:4].values, index=range(22), columns=["boy", "kilo", "yas"])
#print(sonuc2)

#cinsiyet = veriler.iloc[:, -1].values
#print(cinsiyet)

sonuc3 = pd.DataFrame(data=cinsiyet, index=range(22), columns=["cinsiyet"])
#print(sonuc3)

s = pd.concat([sonuc, sonuc2], axis=1)
#print(s)

s2 = pd.concat([s, sonuc3], axis=1)

x_train, x_test, y_train, y_test = train_test_split(s, sonuc3, test_size=0.33, random_state=0)

regressor = LinearRegression()
regressor.fit(x_train, y_train)

y_pred = regressor.predict(x_test)

boy = s2.iloc[:, 3:4].values
#print(boy)

sol = s2.iloc[:, :3]
sag = s2.iloc[:, 4:]

veri = pd.concat([sol, sag], axis=1)

x_train, x_test, y_train, y_test = train_test_split(veri, boy, test_size=0.33, random_state=0)

regressor = LinearRegression()
regressor.fit(x_train, y_train)

y_pred = regressor.predict(x_test)

# Backward Elimination
X = np.append(arr=np.ones((22, 1)).astype(int), values=veri, axis=1)

X_l = veri.iloc[:, [0, 1, 2, 3, 4, 5]].values
X_l = np.array(X_l, dtype=float)
model = sm.OLS(boy, X_l).fit()

print(model.summary())

X_l = veri.iloc[:, [0, 1, 2, 3, 5]].values
X_l = np.array(X_l, dtype=float)
model = sm.OLS(boy, X_l).fit()

print(model.summary())

X_l = veri.iloc[:, [0, 1, 2, 4, 5]].values
X_l = np.array(X_l, dtype=float)
model = sm.OLS(boy, X_l).fit()

print(model.summary())




