# -*- coding: utf-8 -*-
"""
Created on Sat Sep 19 03:59:26 2026

@author: Barış
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier

veriler = pd.read_csv("../data/veriler.csv")

x = veriler.iloc[:, 1:4].values
y = veriler.iloc[:, 4:].values

le = preprocessing.LabelEncoder()

y = le.fit_transform(y.ravel())

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=1)

sc = StandardScaler()

X_train = sc.fit_transform(x_train) 
X_test = sc.transform(x_test)

rfc = RandomForestClassifier(criterion="entropy", n_estimators=10)
rfc.fit(X_train, y_train)

y_pred_rfc = rfc.predict(X_test)

cm = confusion_matrix(y_test, y_pred_rfc)
print(cm)





