#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 28 09:59:08 2020

@author: pierre
"""
import time
debut_algo = time.time()
import os 
os.chdir('/Volumes/PierreLGZ/Mes documents/M1/DataMining/TD_Note')
import pandas as pd
data = pd.read_csv('data_avec_etiquettes.txt', delimiter = "\t")
data.head()

obs = 4898424 
Number_data = obs / len(data)
DF = data

for i in range(1,9):
    data = data.append(DF)
    print(i)

lost = obs - len(data)
data = data.append(DF.head(lost))

data = data.reset_index(drop=True)
print('Finie en',time.time()-debut_algo,'secondes')