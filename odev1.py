# -*- coding: utf-8 -*-
"""
Created on Sat Aug 29 07:15:53 2026

@author: Barış
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm

veriler = pd.read_csv("data/odev_tenis.csv")

veriler2 = veriler.iloc[:, [0, 3, 4]].apply(preprocessing.LabelEncoder().fit_transform) # Kolay yol

# Label Encoder
outlook = veriler.iloc[:, 0:1].values.copy()
#print(outlook)

le = preprocessing.LabelEncoder()
outlook[:, 0] = le.fit_transform(veriler.iloc[:, 0])
#print(outlook)

windy = veriler.iloc[:, 3].values.copy()
windy = le.fit_transform(windy)
#print(windy)

play = veriler.iloc[:, 4].values.copy()
play = le.fit_transform(play)
#print(play)

# One Hot Encoder
ohe = preprocessing.OneHotEncoder()
outlook = ohe.fit_transform(outlook).toarray()
#print(outlook)

# Veri Birlestirme
sonuc = pd.DataFrame(data=outlook, index=range(14), columns=["overcast", "rainy", "sunny"])
#print(sonuc)

humidity = pd.DataFrame(data=veriler.humidity, index=range(14), columns=["humidity"])
#print(humidity)

sonuc2 = pd.DataFrame(data=veriler.iloc[:, 1:2].values, index=range(14), columns=["temperature"])
#print(sonuc2)

sonuc3 = pd.DataFrame(data=windy, index=range(14), columns=["windy"])
#print(sonuc3)

sonuc4 = pd.DataFrame(data=play, index=range(14), columns=["play"])
#print(sonuc4)

s = pd.concat([sonuc, sonuc2, sonuc3, sonuc4], axis=1)
print(s)

x_train, x_test, y_train, y_test = train_test_split(s, humidity, test_size=0.33, random_state=0)

regressor = LinearRegression()
regressor.fit(x_train, y_train)

y_pred = regressor.predict(x_test)

X_l = s.values
X_l = np.array(X_l, dtype=float)
model = sm.OLS(humidity, X_l).fit()
print(model.summary())

X_l = s.iloc[:, [0, 1, 2, 3, 5]].values
X_l = np.array(X_l, dtype=float)
model = sm.OLS(humidity, X_l).fit()

print(model.summary())

s2 = pd.concat([sonuc, humidity, sonuc2, sonuc3], axis=1)
X_l = s2.values
X_l = np.array(X_l, dtype=float)
model = sm.OLS(play, X_l).fit()

print(model.summary())

X_l = s2.iloc[:, [0, 1, 2, 3, 5]].values
X_l = np.array(X_l, dtype=float)
model = sm.OLS(play, X_l).fit()

print(model.summary())

X_l = s2.iloc[:, [0, 1, 2, 5]].values
X_l = np.array(X_l, dtype=float)
model = sm.OLS(play, X_l).fit()

print(model.summary())


X_l = s2.iloc[:, [0, 1, 2]].values
X_l = np.array(X_l, dtype=float)
model = sm.OLS(play, X_l).fit()

print(model.summary())

