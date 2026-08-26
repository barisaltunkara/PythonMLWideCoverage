# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

"""
Prediction -> Modelden öğrenerek belli girdiye göre sonuç tahmini gerçekleştirme. 
Geçmiş ve aradaki tahminleri de yapar. Eksik veri tahmini de yapar.

Forecasting -> Zaman serisinde bir yere kadar yapılan tahminden sonra hiç ulaşmadığımız bir zamanın tahminini yapma.
Örneklem uzayı dışında tahmin yapma.
"""

# Linear Regression y=ax+b
# y-> Bağımlı Değişken x-> Bağımsız Değişken a-> Katsayı

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

# Scipy Linear Regression
df = pd.read_csv("data/satisverileri.csv")
m, b, r, p, err = stats.linregress(df.Aylar, df.Satislar)
x = range(np.min(df.Aylar), np.max(df.Aylar))
y = m*x + b

plt.plot(x, y, "r")
plt.scatter(df.Aylar, df.Satislar)
plt.show()

# Kolon Ayırma
aylar = df[["Aylar"]].values
satislar = df[["Satislar"]].values

x_train, x_test, y_train, y_test = train_test_split(aylar, satislar, test_size=0.33, random_state=0)

sc = StandardScaler()

X_train = sc.fit_transform(x_train)
X_test = sc.fit_transform(x_test)
Y_train = sc.fit_transform(y_train)
Y_test = sc.fit_transform(y_test)

# Sklearn Linear Regression
lr = LinearRegression()
lr.fit(X_train, Y_train)

tahmin = lr.predict(X_test)

plt.scatter(Y_test, lr.predict(X_test))
plt.show()




