# -*- coding: utf-8 -*-
"""
Created on Wed Aug 26 06:14:34 2026

@author: Memocantrolovski
"""

# y = b0 + b1x1 + b2x2 + b3x3 + e

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

veriler = pd.read_csv("data/veriler.csv")

# Label Encoder
ulke = veriler.iloc[:, 0:1].values.copy()
#print(ulke)

le = preprocessing.LabelEncoder()
ulke[:, 0] = le.fit_transform(veriler.iloc[:, 0])
#print(ulke)

cinsiyet = veriler.iloc[:, 4].values
cinsiyet = le.fit_transform(cinsiyet)
#print(cinsiyet)

# One Hot Encoder
ohe = preprocessing.OneHotEncoder()
ulke = ohe.fit_transform(ulke).toarray()
#print(ulke)

# Veri Birlestirme
sonuc = pd.DataFrame(data=ulke, index=range(22), columns=["fr", "tr", "us"])
#print(sonuc)

sonuc2 = pd.DataFrame(data=veriler.iloc[:, 1:4].values, index=range(22), columns=["boy", "kilo", "yas"])
#print(sonuc2)

#cinsiyet = veriler.iloc[:, -1].values
#print(cinsiyet)

sonuc3 = pd.DataFrame(data=cinsiyet, index=range(22), columns=["cinsiyet"])
#print(sonuc3)

s = pd.concat([sonuc, sonuc2], axis=1)
#print(s)

s2 = pd.concat([s, sonuc3], axis=1)





