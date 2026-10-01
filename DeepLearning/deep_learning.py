# -*- coding: utf-8 -*-
"""
Created on Wed Sep 30 03:52:00 2026

@author: Barış
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import preprocessing
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import keras
from keras.models import Sequential
from keras.layers import Dense
from sklearn.metrics import confusion_matrix

veriler = pd.read_csv("../data/Churn_Modelling.csv")

X = veriler.iloc[:, 3:13].values
y = veriler.iloc[:, 13].values

le = preprocessing.LabelEncoder()
X[:, 1] = le.fit_transform(X[:, 1])

le1 = preprocessing.LabelEncoder()
X[:, 2] = le1.fit_transform(X[:, 2])

ohe = ColumnTransformer([("ohe", OneHotEncoder(dtype=float), [1])],
                        remainder="passthrough")

X = ohe.fit_transform(X)
X = X[:, 1:]

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=0)

sc = StandardScaler()

X_train = sc.fit_transform(x_train)
X_test = sc.fit_transform(x_test)

# KERAS
classifier = Sequential()
classifier.add(Dense(6, init="uniform", activation="relu", input_dim=11))
classifier.add(Dense(6, init="uniform", activation="relu"))
classifier.add(Dense(1, init="uniform", activation="sigmoid"))

classifier.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

classifier.fit(X_train, y_train, epoch=50)

y_pred = classifier.predict(X_test)

y_pred = (y_pred > 0.5)

cm = confusion_matrix(y_test, y_pred)

print(cm)
