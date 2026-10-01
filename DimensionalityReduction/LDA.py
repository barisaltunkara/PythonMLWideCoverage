# -*- coding: utf-8 -*-
"""
Created on Thu Oct  1 06:53:21 2026

@author: Barış
"""

"""
Linear Discriminant Analysis

- PCA'den farklı olarak sınıflar arasındaki ayrımı önemser ve maksimize etmeye çalışır.

- PCA bu açıdan gözetimsiz (unsupervised) LDA ise gözetimli (supervised) özelliktedir.
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix

veriler = pd.read_csv("../data/veriler.csv")
X = veriler.iloc[:, 0:13].values
y = veriler.iloc[:, 13].values

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=0)

sc = StandardScaler()

X_train = sc.fit_transform(x_train)
X_test = sc.fit_transform(x_test)

# LDA
lda = LDA(n_components=2)

X_train2 = lda.fit_transform(X_train, y_train)
X_test2 = lda.transform(X_test)

# LDA olmadan logistic regression
classifier = LogisticRegression(random_state=0)
classifier.fit(X_train, y_train)

# LDA dönüşümünden sonra logistic regression
classifier2 = LogisticRegression(random_state=0)
classifier2.fit(X_train2, y_train)

# Tahminler
y_pred = classifier.predict(X_test)

y_pred2 = classifier2.predict(X_test2)

print("LDA olmadan:")
cm = confusion_matrix(y_test, y_pred)
print(cm)

print("LDA dönüşümlü:")
cm1 = confusion_matrix(y_test, y_pred2)
print(cm1)

print("LDA'siz ve LDA'li:")
cm2 = confusion_matrix(y_pred, y_pred2)
print(cm2)



