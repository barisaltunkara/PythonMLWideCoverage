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

"""with open("../data/restoran.csv", "r") as df:
    df = df.read()
    df = df.replace(", ", " ")

with open("../data/yeni_restoran.csv", "w") as output:
    output.write(df)
"""

yorumlar = pd.read_csv("../data/yeni_restoran.csv")

yorum = re.sub("[^a-zA-Z]", " ", yorumlar["Review"][0])
yorum = yorum.lower()
yorum = yorum.split()

durma = nltk.download("stopwords")

ps = PorterStemmer()


