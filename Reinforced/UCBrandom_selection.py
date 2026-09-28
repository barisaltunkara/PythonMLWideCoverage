# -*- coding: utf-8 -*-
"""
Created on Mon Sep 28 04:02:13 2026

@author: Barış
"""

# Upper Confidence Bound - reinforced learning
# Rastgele seçimin gösterimi. UCB ile kıyaslanması yapılacak

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
import math

veriler = pd.read_csv("../data/Ads_CTR_Optimisation.csv")

# Random selection
N = 10000 # tıklama sayısı
d = 10 # ilan sayısı
toplam = 0
secilenler = []

for n in range(0, N):
    ad = random.randrange(d)
    secilenler.append(ad)
    odul = veriler.values[n, ad] # Verilerdeki n. satır = 1 ise ödül 1 değilse 0
    toplam += odul

plt.hist(secilenler)
plt.show()

"""
UCB ALGORİTMASI

Adım 1: Her turda (tur sayısı n olsun), her reklam alternatifi (i için) aşağıdaki sayılar tutulur:
    - Ni(n): i sayılı reklamın o ana kadarki tıklama sayısı
    - Ri(n): o ana kadarki i reklamından gelen toplam ödül
    
Adım 2: Yukarıdaki bu iki sayıdan, aşağıdaki değerler hesaplanır:
    - O ana kadarki her reklamın ortalama ödülü Ri(n)/Ni(n)
    - Güven aralığı için aşağı ve yukarı oynama potansiyeli di(n) = sqrt(3/2 * log(n)/Ni(n))
    
Adım 3: En yüksek UCB değerine sahip olanı alırız.
"""

# UCB
tiklamalar = [0] * d # o ana kadarki tıklamalar Ni(n)
oduller = [0] * d # ilk başta bütün ilanların ödülü 0 Ri(n)
toplam1 = 0 # toplam ödül
secilenler = []

for n in range(0, N):
    ad = 0 # seçilen ilan
    max_ucb = 0
    
    for i in range(0, d):
        if tiklamalar[i] > 0:
            ortalama = oduller[i] / tiklamalar[i]
            delta = math.sqrt((3/2) * math.log(n)/tiklamalar[i])
            ucb = ortalama + delta
        else:
            ucb = N*10
            
        if max_ucb < ucb: # max tan büyük bir ucb
            max_ucb = ucb
            ad = i
            
    
    odul = veriler.values[n, ad]
    secilenler.append(ad)
    tiklamalar[ad] += 1
    oduller[ad] += odul
    toplam1 = toplam1 + odul

print(f"Toplam Ödül: {toplam1}")

plt.hist(secilenler)
plt.show()

