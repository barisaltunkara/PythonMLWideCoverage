# -*- coding: utf-8 -*-
"""
Created on Sun Aug 30 10:45:54 2026

@author: Barış
"""

from sklearn.tree import DecisionTreeRegressor
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

veriler = pd.read_csv("data/maaslar.csv")
x = veriler.iloc[:, 1:2].values
y = veriler.iloc[:, 2:].values

r_dt = DecisionTreeRegressor(random_state=0)
r_dt.fit(x, y)

plt.scatter(x, y)
plt.plot(x, r_dt.predict(x), "r")
plt.show()

print(r_dt.predict([[11]]))
print(r_dt.predict([[6.6]]))

z = x + 0.5
k = x - 0.4

plt.scatter(x, y)
plt.plot(x, r_dt.predict(x), "r")
plt.plot(x, r_dt.predict(z), "g")
plt.plot(x, r_dt.predict(k), "y")
plt.show()


