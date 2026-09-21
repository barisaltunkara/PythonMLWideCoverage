# -*- coding: utf-8 -*-
"""
Created on Sat Sep 19 06:38:19 2026

@author: Barış
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix, roc_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier

veriler = pd.read_excel("../data/Iris.xls")

x = veriler.iloc[:, 0:4].values
y = veriler.iloc[:, 4:].values

le = preprocessing.LabelEncoder()

y = le.fit_transform(y.ravel())

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=0)

sc = StandardScaler()

X_train = sc.fit_transform(x_train) 
X_test = sc.transform(x_test) 

# Logistic Regression
log_reg = LogisticRegression(random_state=0)
log_reg.fit(X_train, y_train)

y_pred_log = log_reg.predict(X_test)

cm = confusion_matrix(y_test, y_pred_log)
print("Logistic: \n", cm)

# KNN
knn = KNeighborsClassifier(n_neighbors=3, metric="minkowski")
knn.fit(X_train, y_train)

y_pred_knn = knn.predict(X_test)

cm = confusion_matrix(y_test, y_pred_knn)
print("KNN: \n", cm)

# Naive-Bayes
gnb = GaussianNB()
gnb.fit(X_train, y_train)

y_pred_gnb = gnb.predict(X_test)

cm = confusion_matrix(y_test, y_pred_gnb)
print("Naive-Bayes: \n", cm)

# Decision Tree
tree = DecisionTreeClassifier(criterion="entropy")
tree.fit(X_train, y_train)

y_pred_tree = tree.predict(X_test)

cm = confusion_matrix(y_test, y_pred_tree)
print(cm)

# Random Forest
rfc = RandomForestClassifier(criterion="entropy", n_estimators=10, random_state=1)
rfc.fit(X_train, y_train)

y_pred_rfc = rfc.predict(X_test)

cm = confusion_matrix(y_test, y_pred_rfc)
print("Random Forest: \n", cm)

# SVC
svc = SVC(kernel="linear")
svc.fit(X_train, y_train)

y_pred_svc = svc.predict(X_test)

cm = confusion_matrix(y_test, y_pred_svc)
print("Linear: \n", cm)

svc = SVC(kernel="rbf")
svc.fit(X_train, y_train)

y_pred_svc = svc.predict(X_test)

cm = confusion_matrix(y_test, y_pred_svc)
print("RBF: \n", cm)

svc = SVC(kernel="sigmoid")
svc.fit(X_train, y_train)

y_pred_svc = svc.predict(X_test)

cm = confusion_matrix(y_test, y_pred_svc)
print("Sigmoid: \n", cm)

svc = SVC(kernel="poly")
svc.fit(X_train, y_train)

y_pred_svc = svc.predict(X_test)

cm = confusion_matrix(y_test, y_pred_svc)
print("Polynomial: \n", cm)






