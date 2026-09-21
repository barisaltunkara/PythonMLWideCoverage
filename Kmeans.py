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