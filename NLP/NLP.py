# -*- coding: utf-8 -*-
"""
Created on Mon Sep 28 08:08:26 2026

@author: Barış
"""

import numpy as np
import pandas as pd
import re
import nltk
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix

"""
with open("../data/restoran.csv", "r") as df:
    df = df.read()
    df = df.replace(", ", " ")

with open("../data/yeni_restoran.csv", "w") as output:
    output.write(df)
"""

yorumlar = pd.read_csv("../data/yeni_restoran.csv")
yorumlar = yorumlar.dropna().reset_index(drop=True)

yorum = re.sub("[^a-zA-Z]", " ", yorumlar["Review"][0])
yorum = yorum.lower()
yorum = yorum.split()

durma = nltk.download("stopwords")

ps = PorterStemmer()

yorum = [ps.stem(kelime) for kelime in yorum if not kelime in set(stopwords.words("english"))]

yorum = " ".join(yorum)

# Preprocessing
derleme = []
for i in range(yorumlar.shape[0]):
    yorum = re.sub("[^a-zA-Z]", " ", yorumlar["Review"][i])
    yorum = yorum.lower()
    yorum = yorum.split()
    yorum = [ps.stem(kelime) for kelime in yorum if not kelime in set(stopwords.words("english"))]
    yorum = " ".join(yorum)
    
    derleme.append(yorum)


# Feature extraction (Bag of Words - BOW)
cv = CountVectorizer(max_features=1000) # En fazla kullanılacak kelime sayısı max_features

X = cv.fit_transform(derleme).toarray() # Sparse matrix
y = yorumlar.Liked.values

# Makine Öğrenmesi
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=1)

gnb = GaussianNB()
gnb.fit(X_train, y_train)
y_pred = gnb.predict(X_test)

cm = confusion_matrix(y_test, y_pred)
print(cm)


