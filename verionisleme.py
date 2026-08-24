# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

# kutuphaneler
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# kod bolumu
# veri yukleme
veriler = pd.read_csv("veriler.csv")
#print(veriler.describe())


# veri on isleme
eksikveriler = pd.read_csv("eksikveriler.csv")
#print(eksikveriler.describe())

# Eksik veri doldurma
imputer = SimpleImputer(missing_values=np.nan, strategy="median") # mean, most_frequent, constant

Yas = eksikveriler.iloc[:, 1:4].values
#print(Yas)

imputer = imputer.fit(Yas[:, 1:4])
Yas[:, 1:4] = imputer.transform(Yas[:, 1:4])
#print(Yas)

# Label Encoder
ulke = veriler.iloc[:, 0:1].values
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

sonuc2 = pd.DataFrame(data=Yas, index=range(22), columns=["boy", "kilo", "yas"])
#print(sonuc2)

#cinsiyet = veriler.iloc[:, -1].values
#print(cinsiyet)

sonuc3 = pd.DataFrame(data=cinsiyet, index=range(22), columns=["cinsiyet"])
#print(sonuc3)

s = pd.concat([sonuc, sonuc2], axis=1)
#print(s)

s2 = pd.concat([s, sonuc3], axis=1)


# Train-Test Ayirimi
x_train, x_test, y_train, y_test = train_test_split(s, sonuc3, test_size=0.33, random_state=0)

# Veri Olceklendirme
sc = StandardScaler()

X_train = sc.fit_transform(x_train)
X_test = sc.fit_transform(x_test)





