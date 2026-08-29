# -*- coding: utf-8 -*-
"""
Created on Sat Aug 29 09:20:52 2026

@author: Barış
"""

# y = b0 + b1x + b1x^2 + ... + bhx^h + e

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm

maaslar = pd.read_csv("data/maaslar.csv")

plt.plot(maaslar["Egitim Seviyesi"], maaslar.maas)
plt.show()



