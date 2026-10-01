# -*- coding: utf-8 -*-
"""
Created on Thu Oct  1 10:34:49 2026

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
from sklearn.model_selection import GridSearchCV

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

# Parametre optimizasyonu ve algoritma seçimi

p = [{"C": [1, 2, 3, 4, 5], "kernel": ["linear", "rbf"]},
     {"C": [1, 10, 100, 1000], "kernel": ["rbf"], "gamma": [1, 0.5, 0.1, 0.01, 0.001]}]

"""
Grid Search parametreleri:
    estimator: sınıflandırma algoritması (optimize etmek istediğimiz)
    param_grid: parametreler/denenecekler
    scoring: neye göre skorlanacak (örn. accuracy)
    cv: kaç katlamalı olacağı
    n_jobs: aynı anda çalışacak işler
"""

grids = GridSearchCV(estimator = classifier, 
                     param_grid = p,
                     scoring = "accuracy",
                     cv = 10,
                     n_jobs=-1)

grid_search = grids.fit(X_train, y_train)

eniyisonuc = grid_search.best_score_

eniyiparametreler = grid_search.best_params_

print(eniyisonuc)
print(eniyiparametreler)




