# -*- coding: utf-8 -*-
"""
Created on Thu Oct  1 06:01:30 2026

@author: Barış
"""

"""
Principal Component Analysis
PCA Algoritması:
    
- İndirgenmek istenen boyut k olsun
- Veriyi Standartlaştır
- Covariance veya Correlation matrisinden eigen değerleri ve eigen vektörleri elde et. Veya SVD kullan.
- Eigen değerleri büyükten küçüğe sırala ve k tanesini al.
- Seçilen k eigen değerden W projeksiyon matrisini oluştur.
- Orijinal veri kümesi X'i W kullanarak dönüştür ve k-boyutlu Y uzayını elde et.
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix

veriler = pd.read_csv("../data/veriler.csv")
X = veriler.iloc[:, 0:13].values
y = veriler.iloc[:, 13].values

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=0)

sc = StandardScaler()

X_train = sc.fit_transform(x_train)
X_test = sc.fit_transform(x_test)

# PCA
pca = PCA(n_components=2)

X_train2 = pca.fit_transform(X_train)
X_test2 = pca.transform(X_test)

# PCA olmadan logistic regression
classifier = LogisticRegression(random_state=0)
classifier.fit(X_train, y_train)

# PCA dönüşümünden sonra logistic regression
classifier2 = LogisticRegression(random_state=0)
classifier2.fit(X_train2, y_train)

# Tahminler
y_pred = classifier.predict(X_test)

y_pred2 = classifier2.predict(X_test2)

print("PCA olmadan:")
cm = confusion_matrix(y_test, y_pred)
print(cm)

print("PCA dönüşümlü:")
cm1 = confusion_matrix(y_test, y_pred2)
print(cm1)

print("PCA'siz ve PCA'li:")
cm2 = confusion_matrix(y_pred, y_pred2)
print(cm2)
