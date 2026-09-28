# -*- coding: utf-8 -*-
"""
Created on Mon Sep 28 05:42:18 2026

@author: Barış
"""

"""
Thompson Örneklemesi

Adım 1: Her aksiyon için aşağıdaki iki sayıyı hesapla:
    - Ni1(n): O ana kadar ödül olarak 1 gelme sayısı
    - Ni0(n): O ana kadar ödül olarak 0 gelme sayısı
    
Adım 2: Her ilan için aşağıda verilen Beta dağılımında bir rastgele sayı üretiyoruz.
Tetai(N) = Beta(Ni1(n) + 1,Ni0(n) + 1)

Adım 3: En yüksek beta değerine sahip olanı seçiyoruz.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
import math

veriler = pd.read_csv("../data/Ads_CTR_Optimisation.csv")

# Thompson
N = 10000 # tıklama sayısı
d = 10 # ilan sayısı
birler = [0] * d # Ni1
sifirlar = [0] * d # Ni0
toplam = 0 # toplam ödül
secilenler = []

for n in range(0, N):
    ad = 0 # seçilen ilan
    max_th = 0
    
    for i in range(0, d):
        rasbeta = random.betavariate(birler[i] + 1, sifirlar[i] + 1)
        
        if max_th < rasbeta: # max tan büyük bir ucb
            max_th = rasbeta
            ad = i
    
    odul = veriler.values[n, ad]
    if odul == 1:
        birler[ad] += 1
    else:
        sifirlar[ad] += 1
    secilenler.append(ad)
    toplam = toplam + odul

print(f"Toplam Ödül: {toplam}")

plt.hist(secilenler)
plt.show()

