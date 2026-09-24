# -*- coding: utf-8 -*-
"""
Created on Wed Sep 23 21:45:33 2026

@author: Barış
"""

# Agglomerative Clustering

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, AgglomerativeClustering
import scipy.cluster.hierarchy as sch

veriler = pd.read_csv("data/musteriler.csv")

X = veriler.iloc[:, 3:].values

hc = AgglomerativeClustering(n_clusters=3, linkage="ward")
y_pred = hc.fit_predict(X)

plt.scatter(X[y_pred==0, 0], X[y_pred==0, 1], s=100, c="red")
plt.scatter(X[y_pred==1, 0], X[y_pred==1, 1], s=100, c="green")
plt.scatter(X[y_pred==2, 0], X[y_pred==2, 1], s=100, c="blue")
plt.show()

plt.hist(y_pred)
plt.show()

# Dendrogram
dendrogram = sch.dendrogram(sch.linkage(X, method="ward"))
plt.show()

