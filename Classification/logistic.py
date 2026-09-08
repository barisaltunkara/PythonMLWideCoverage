# -*- coding: utf-8 -*-
"""
Created on Tue Sep  8 14:12:30 2026

@author: Barış
"""

"""
sigma(t) = e^t / (e^t + 1) = 1 / (1 + e^(-t))

t = beta0 + beta1*x t = A + B*x

p(x) = 1 / (1 + e^(- (beta0 + beta1*x)))

beta0 + beta1*x1 + beta2*x2 + ... + betam*xm = beta0 + sum(betai*xi)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn import preprocessing

veriler = pd.read_csv("../data/veriler.csv")

x = veriler.iloc[:, 1:4].values
y = veriler.iloc[:, 4:].values

le = preprocessing.LabelEncoder()

y = le.fit_transform(y)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=0)

sc = StandardScaler()

X_train = sc.fit_transform(x_train) # fit transform eğitim verisinden öğreniyor 
X_test = sc.transform(x_test) # dolayısıyla test verisinde transform kullanmak yeterli

# Logistic Regression
log_reg = LogisticRegression(random_state=0)
log_reg.fit(X_train, y_train)

y_pred = log_reg.predict(X_test)


