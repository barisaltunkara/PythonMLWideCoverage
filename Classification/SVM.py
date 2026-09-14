#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep  8 19:15:40 2026

@author: barisaltunkara
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix
from sklearn.svm import SVC

veriler = pd.read_csv("../data/veriler.csv")

x = veriler.iloc[:, 1:4].values
y = veriler.iloc[:, 4:].values

le = preprocessing.LabelEncoder()

y = le.fit_transform(y.ravel())

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=1)

sc = StandardScaler()

X_train = sc.fit_transform(x_train) 
X_test = sc.transform(x_test)

svc = SVC(kernel="linear")
svc.fit(X_train, y_train)

y_pred_svc = svc.predict(X_test)

cm = confusion_matrix(y_test, y_pred_svc)
print(cm)

svc = SVC(kernel="rbf")
svc.fit(X_train, y_train)

y_pred_svc = svc.predict(X_test)

cm = confusion_matrix(y_test, y_pred_svc)
print(cm)

svc = SVC(kernel="sigmoid")
svc.fit(X_train, y_train)

y_pred_svc = svc.predict(X_test)

cm = confusion_matrix(y_test, y_pred_svc)
print(cm)

svc = SVC(kernel="poly")
svc.fit(X_train, y_train)

y_pred_svc = svc.predict(X_test)

cm = confusion_matrix(y_test, y_pred_svc)
print(cm)


# KERNEL TRICK

# gi(xj) = exp((-||xj - ui||^2) / 2*sigmai^2)








