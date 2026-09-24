# -*- coding: utf-8 -*-
"""
Created on Wed Sep 23 22:22:09 2026

@author: Barış
"""

""" 
Birliktelik Kural Çıkarımı (Association Rule Mining)

Support = actions that include a / total action number

Confidence(a->b) = actions that include a and b together / actions that include a

Lift(a->b) = Confidence(a->b)/Support(b)

If lift is bigger than 1, we can say that taking a increases probability of taking b
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from apyori import apriori # ymoch/apyori Github file

veriler = pd.read_csv("data/sepet.csv", header=None)

t = []
for i in range(0, 7501):
    t.append([str(veriler.values[i, j]) for j in range(0, 20)])

kurallar = apriori(t, min_support=0.01, min_lift=3, min_length=2)

print(list(kurallar))

# Diğer bir algoritma Eclat Algoritması