#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep  8 16:25:40 2026

@author: barisaltunkara
"""

"""
Accuracy M, acc(M): model M için yüzde kaç doğru sınıflandırma olduğudur.
- Error rate (missclassification rate) = 1 - acc(M)
- Alternatif ölçümler (e.g., for cancer diagnosis)

sensitivity = t-pos/(t-pos + f-neg) true positive recognition rate
specificity = t-neg/(t-neg + f-pos) true negative recognition rate
precision = t-pos/(t-pos + f-pos)
accuracy = (t-pos + t-neg) / total
"""

import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix

df = pd.DataFrame({"classes": ["buy_computer = yes", "buy_computer = no", 
                               "total"],
                   "buy_computer = yes": [6954, 412, 7366],
                   "buy_computer = no": [46, 2588, 2634]})

df["total"] = df.drop(["classes"], axis=1).sum(axis=1)
df["recognition(%)"] = np.array([100*6954/7000, 
                                 100*2588/3000,
                                 100*(6954+2588)/10000])

conf_mat = pd.DataFrame(columns=["C1", "C2"])
conf_mat.loc["C1"] = ["True positive", "False Negative"]
conf_mat.loc["C2"] = ["False positive", "True Negative"]

print(df)
print(conf_mat)




