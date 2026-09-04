# -*- coding: utf-8 -*-
"""
Created on Fri Sep  4 11:42:26 2026

@author: Barış
"""
"""Ensemble Learning
- Boosting
- Bagging
- Random Forest - Çoklu karar ağaçlarının kombine çalışması ile oluşur - Majority Vote
- AdaBoost
- Stacking
- Blending
- MAVL

Diverse Predictors:
    - Logistic Regression
    - SVM Classifier
    - Random Forest Classifier
    - Others
    
Combination of weak classifiers

Regresyonda karar ağaçlarının tahminlerinin ortalaması çıktı olarak alınır.
Karar ağaçlarında veri arttıkça overfittingde bir artış gözlemlenmekte. Ek olarak
çok fazla dallanmaya neden olabileceği için sığ ve zayıf karar ağaçlarının kombinasyonu kullanılır.
"""

from sklearn.ensemble import RandomForestRegressor
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

veriler = pd.read_csv("data/maaslar.csv")
x = veriler.iloc[:, 1:2].values
y = veriler.iloc[:, 2:].values

rf_reg = RandomForestRegressor(random_state=0, n_estimators=10)
rf_reg.fit(x, y.ravel())

plt.scatter(x, y)
plt.plot(x, rf_reg.predict(x), "r")
plt.show()

z = x + 0.5
k = x - 0.4

plt.scatter(x, y)
plt.plot(x, rf_reg.predict(x), "r")
plt.plot(x, rf_reg.predict(z), "g")
plt.plot(x, rf_reg.predict(k), "y")
plt.show()

print(rf_reg.predict([[6.5]]))


