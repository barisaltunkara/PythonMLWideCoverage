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








