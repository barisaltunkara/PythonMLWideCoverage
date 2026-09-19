# -*- coding: utf-8 -*-
"""
Created on Thu Sep 17 19:14:23 2026

@author: Barış
"""

"""
Information Gain ID3

Info(D) = -sum(pi*log2(pi))

InfoA(D) = sum((|Dj|/|D|)*I(Dj))

Gain(A) = Info(D) - InfoA(D)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix
from sklearn.tree import DecisionTreeClassifier
import math

def info(a, b):
    n = a + b
    
    if a==0 or b==0:
        return 0
    else:
        return -(a/n)*(math.log2(a/n)) - (b/n)*(math.log2(b/n))


def infoA(a, b, c, list1):
    n = a + b + c
    ainfo = info(list1[0][0], list1[0][1])
    binfo = info(list1[1][0], list1[1][1])
    cinfo = info(list1[2][0], list1[2][1])
    
    return (a/n)*ainfo + (b/n)*binfo + (c/n)*cinfo

print(info(9, 5) - infoA(5, 4, 5, [[2,3],[4,0],[3,2]]))

veriler = pd.read_csv("../data/veriler.csv")

x = veriler.iloc[:, 1:4].values
y = veriler.iloc[:, 4:].values

le = preprocessing.LabelEncoder()

y = le.fit_transform(y.ravel())

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=1)

sc = StandardScaler()

X_train = sc.fit_transform(x_train) 
X_test = sc.transform(x_test)

tree = DecisionTreeClassifier(criterion="entropy")
tree.fit(X_train, y_train)

y_pred_tree = tree.predict(X_test)

cm = confusion_matrix(y_test, y_pred_tree)
print(cm)

# Varsayılan olarak karar ağacı Gini kullanır. Gini log2(pi) yerine direkt pi kullanılarak hesaplanır.



