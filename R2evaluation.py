# -*- coding: utf-8 -*-
"""
Created on Fri Sep  4 17:41:22 2026

@author: Barış
"""

"""R2 - R-Square Yöntemi
Hata Kareleri Toplamı = Topla(yi - y'i)^2
Ortalama Farkların Toplamı = Topla(yi - yort)^2
R^2 = 1 - HKT/OFT

R^2 in problemi eklenen her değişkenin sonucu arttırıcı yönde etki yapıyor olması.
Negatif etkileyen bir parametre 0 olarak görüldüğü için sonuca etki yapmaz.

Bu problem için Adjusted R^2 yöntemi kullanılıyor.

Adjusted R^2 = 1 - (1 - R^2) * (n - 1) / (n - p - 1)
n -> eleman sayısı
p -> değişken sayısı
"""

from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

veriler = pd.read_csv("data/maaslar.csv")
x = veriler.iloc[:, 1:2].values
y = veriler.iloc[:, 2:].values

df = pd.DataFrame(columns=["R^2"])

# Random Forest Regression
rf_reg = RandomForestRegressor(random_state=0, n_estimators=10)
rf_reg.fit(x, y.ravel())

y_pred = rf_reg.predict(x)

df.loc["RandomForestx"] = [r2_score(y, y_pred)]
print(f"Random Forest R^2 değeri x: {r2_score(y, y_pred):.2}")

z = x + 0.5
k = x - 0.4

y_pred1 = rf_reg.predict(z)
y_pred2 = rf_reg.predict(k)

df.loc["RandomForestz"] = [r2_score(y, y_pred1)]
df.loc["RandomForestk"] = [r2_score(y, y_pred2)]
print(f"Random Forest R^2 değeri z: {r2_score(y, y_pred1):.2}")
print(f"Random Forest R^2 değeri k: {r2_score(y, y_pred2):.2}")

# Decision Tree Regression
r_dt = DecisionTreeRegressor(random_state=0)
r_dt.fit(x, y)

y_pred_tree = r_dt.predict(x)

df.loc["DecisionTree"] = [r2_score(y, y_pred_tree)]
print(f"Decision Tree R^2 değeri x: {r2_score(y, y_pred_tree):.2}")

# SVR Regression RBF
sc1 = StandardScaler()
x_olcekli = sc1.fit_transform(x)
sc2 = StandardScaler()
y_olcekli = sc2.fit_transform(y)

svr_reg = SVR(kernel="rbf")
svr_reg.fit(x_olcekli, y_olcekli.ravel())

y_pred_SVR = svr_reg.predict(x_olcekli)

df.loc["SVRrbf"] = [r2_score(y_olcekli, y_pred_SVR)]
print(f"SVR rbf R^2 değeri x: {r2_score(y_olcekli, y_pred_SVR):.2}")

# Polynomial Regression 2
poly_reg = PolynomialFeatures(degree=2)
x_poly = poly_reg.fit_transform(x)

lin_reg1 = LinearRegression()
lin_reg1.fit(x_poly, y)

y_pred_poly = lin_reg1.predict(x_poly)

df.loc["Poly2"] = [r2_score(y, y_pred_poly)]
print(f"Polynomial (2) R^2 değeri x: {r2_score(y, y_pred_poly):.2}")

# Polynomial Regression 4
poly_reg1 = PolynomialFeatures(degree=4)
x_poly1 = poly_reg1.fit_transform(x)

lin_reg2 = LinearRegression()
lin_reg2.fit(x_poly1, y)

y_pred_poly1 = lin_reg2.predict(x_poly1)

df.loc["Poly4"] = [r2_score(y, y_pred_poly1)]
print(f"Polynomial (4) R^2 değeri x: {r2_score(y, y_pred_poly1):.2}")

#Linear Regression
lr = LinearRegression()
lr.fit(x, y)

tahmin = lr.predict(x)

df.loc["Lineer"] = [r2_score(y, tahmin)]
print(f"Lineer R^2 değeri x: {r2_score(y, tahmin):.2}")

print(df)

