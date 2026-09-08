# -*- coding: utf-8 -*-
"""
Created on Mon Sep  7 13:51:23 2026

@author: Barış
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
import statsmodels.api as sm
from scipy import stats


veriler = pd.read_csv("../data/maaslar_yeni.csv")

x = veriler.iloc[:, 2:5].values
y = veriler.iloc[:, 5:].values


# Lineer Regression
model = sm.OLS(y, x)
#print(model.fit().summary())

lin_reg = LinearRegression()
lin_reg.fit(x, y)

model1 = sm.OLS(lin_reg.predict(x), x)
#print(model1.fit().summary())

x1 = veriler.iloc[:, 2:3].values

model2 = sm.OLS(y, x1)
#print(model2.fit().summary())

lin_reg1 = LinearRegression()
lin_reg1.fit(x1, y)

model3 = sm.OLS(lin_reg1.predict(x1), x1)
#print(model3.fit().summary())


# Polynomial Regression
poly_reg = PolynomialFeatures(degree=2)
x_poly = poly_reg.fit_transform(x)

lin_reg2 = LinearRegression()
lin_reg2.fit(x_poly, y)

y_pred_poly = lin_reg2.predict(x_poly)

model4 = sm.OLS(y_pred_poly, x)
#print(model4.fit().summary())

poly_reg1 = PolynomialFeatures(degree=2)
x_poly1 = poly_reg.fit_transform(x1)

lin_reg3 = LinearRegression()
lin_reg3.fit(x_poly1, y)

y_pred_poly1 = lin_reg3.predict(x_poly1)

model5 = sm.OLS(y_pred_poly1, x1)
#print(model5.fit().summary())


# SVR RBF Regression
sc1 = StandardScaler()
x_olcekli = sc1.fit_transform(x)
sc2 = StandardScaler()
y_olcekli = sc2.fit_transform(y)

svr_reg = SVR(kernel="rbf")
svr_reg.fit(x_olcekli, y_olcekli.ravel())

y_pred_SVR = svr_reg.predict(x_olcekli)

model6 = sm.OLS(y_pred_SVR, x_olcekli)
#print(model6.fit().summary())

sc3 = StandardScaler()
x_olcekli1 = sc3.fit_transform(x1)
sc4 = StandardScaler()
y_olcekli1 = sc4.fit_transform(y)

svr_reg1 = SVR(kernel="rbf")
svr_reg1.fit(x_olcekli1, y_olcekli1.ravel())

y_pred_SVR1 = svr_reg1.predict(x_olcekli1)

model7 = sm.OLS(y_pred_SVR1, x_olcekli1)
#print(model7.fit().summary())


# Decision Tree Regression
r_dt = DecisionTreeRegressor(random_state=0)
r_dt.fit(x, y)

y_pred_tree = r_dt.predict(x)
model8 = sm.OLS(y_pred_tree, x)
#print(model8.fit().summary())

r_dt1 = DecisionTreeRegressor(random_state=0)
r_dt1.fit(x1, y)

y_pred_tree1 = r_dt1.predict(x1)
model9 = sm.OLS(y_pred_tree1, x1)
#print(model9.fit().summary())


# Random Forest Regression
rf_reg = RandomForestRegressor(random_state=0, n_estimators=10)
rf_reg.fit(x, y.ravel())

y_pred_rf = rf_reg.predict(x)
model10 = sm.OLS(y_pred_rf, x)
print(model10.fit().summary())

rf_reg1 = RandomForestRegressor(random_state=0, n_estimators=10)
rf_reg1.fit(x1, y.ravel())

y_pred_rf1 = rf_reg1.predict(x1)
model11 = sm.OLS(y_pred_rf1, x1)
print(model11.fit().summary())


print(veriler.drop(["unvan"], axis=1).corr())
