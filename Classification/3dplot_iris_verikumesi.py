# -*- coding: utf-8 -*-
"""
Created on Sat Sep 19 07:28:48 2026

@author: Barış
"""

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn import datasets

# import iris dataset
iris = datasets.load_iris()

X = iris.data[:, :2]
y = iris.target

x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5

plt.figure(2, figsize=(8, 6))
plt.clf()

# Plot the training points
plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Set1, edgecolors="k")
plt.xlabel("Sepal length")
plt.ylabel("Sepal width")

plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
plt.xticks(())
plt.yticks(())
plt.show()

fig = plt.figure(1, figsize=(8, 6))
ax = Axes3D(fig, elev=-150, azim=110)
ax.scatter(iris.data[:, 0], iris.data[:, 1], iris.data[:, 2],
           c=y, cmap=plt.cm.Set1, edgecolor="k", s=40)
ax.set_title("IRIS verisi")
ax.set_xlabel("birinci özellik")
ax.xaxis.set_ticklabels([])
ax.set_ylabel("ikinci özellik")
ax.yaxis.set_ticklabels([])
ax.set_zlabel("üçüncü özellik")
ax.zaxis.set_ticklabels([])

plt.show()



