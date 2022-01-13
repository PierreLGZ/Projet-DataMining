#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 17 18:28:45 2020

@author: pierre
"""

import time
debut_algo = time.time()
import pandas as pd
import pickle
import os
os.chdir('/Volumes/PierreLGZ/Mes documents/M1/DataMining/TD_Note')
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn import metrics

col = ['V170', 'V190', 'V189', 'V188', 'V187', 
        'V185', 'V181', 'V193', 'V195', 'V168', 
        'V166', 'V162', 'V160', 'V169', 'V159', 
        'V165', 'V167']#17

data_TD = pd.read_csv('data_avec_etiquettes.txt', delimiter = "\t", usecols = col)

#On standardise les données quantitatives
num_cols = data_TD._get_numeric_data().columns
standard = StandardScaler()
Data_array = standard.fit_transform(data_TD[num_cols])
data_TD[num_cols] = pd.DataFrame(Data_array)
#On transforme les données qualitatives
categ_cols = list(set(col) - set(num_cols))
label_encoder = LabelEncoder()
data_TD[categ_cols] = data_TD[categ_cols].apply(label_encoder.fit_transform)

#On charge le modèle
pkl_filename = "modele_scoring.pkl"
with open(pkl_filename, 'rb') as file:
    model_lda = pickle.load(file)

#On l'applique
data_TD['Score'] = model_lda.predict_proba(data_TD)[:,1]
data_TD.to_csv("scores.txt", columns =['Score'], sep = "\t")

#On recupere les vraie données cible
data_TD.loc[data['V200'] != 'm16', 'V200_m16'] = 0
data_TD.loc[data['V200'] == 'm16', 'V200_m16'] = 1

#On compare sur les 10 000 premières valeures
Final = data_TD.sort_values(by = 'Score', ascending = False).head(10000)
# 1.0    7956
# 0.0    2044

print('Finie en',time.time()-debut_algo,'secondes')#89.8s/80
