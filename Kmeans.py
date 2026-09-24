# -*- coding: utf-8 -*-
"""
Created on Tue Sep 22 01:00:36 2026

@author: Barış
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

veriler = pd.read_csv("data/musteriler.csv")

X = veriler.iloc[:, 3:].values

kmeans = KMeans(n_clusters=3, init="k-means++", random_state=0)
kmeans.fit(X)

print(kmeans.cluster_centers_)

sonuclar = []
for i in range(1, 10):
    kmeans = KMeans(n_clusters=i, init="k-means++", random_state=0)
    kmeans.fit(X)
    sonuclar.append(kmeans.inertia_) # inertia bize WCSS verilerini verir
    
plt.plot(range(1, 10), sonuclar)
plt.show()

kmeans = KMeans(n_clusters=4, init="k-means++", random_state=0)
y_pred = kmeans.fit_predict(X)

plt.scatter(X[y_pred==0, 0], X[y_pred==0, 1], s=100, c="red")
plt.scatter(X[y_pred==1, 0], X[y_pred==1, 1], s=100, c="green")
plt.scatter(X[y_pred==2, 0], X[y_pred==2, 1], s=100, c="blue")
plt.scatter(X[y_pred==3, 0], X[y_pred==3, 1], s=100, c="yellow")
plt.show()

