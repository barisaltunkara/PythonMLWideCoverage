# -*- coding: utf-8 -*-
"""
Created on Thu Oct  1 10:24:43 2026

@author: Barış
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score

veriler = pd.read_csv("../data/Social_Network_Ads.csv")
X = veriler.iloc[:, [2, 3]].values
y = veriler.iloc[:, 4].values

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

sc = StandardScaler()

X_train = sc.fit_transform(x_train)
X_test = sc.fit_transform(x_test)

classifier = SVC(kernel="rbf", random_state=0)
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)

cm = confusion_matrix(y_test, y_pred)
print(cm)

"""
cross_val_score parameters:
    1. estimator: classifier 
    2. X
    3. y
    4. cv: kaç katmanlı
"""

basari = cross_val_score(classifier, X=X_train, y=y_train, cv=4)
print(basari.mean())
print(basari.std())

