#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 17 18:26:16 2020

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

col = ['V160', 'V187', 'V163', 'V193']
data_TD = pd.read_csv('data_avec_etiquettes.txt', delimiter = "\t",usecols = col)

#On standardise les variables quantitatives
num_cols = data_TD._get_numeric_data().columns
standard = StandardScaler()
Data_array = standard.fit_transform(data_TD[num_cols])
data_TD[num_cols] = pd.DataFrame(Data_array)
#On transforme les variables qualitatives
categ_cols = list(set(col) - set(num_cols))
label_encoder = LabelEncoder()
data_TD[categ_cols] = data_TD[categ_cols].apply(label_encoder.fit_transform)

#On recupere le modele
pkl_filename = "modele_classif.pkl"
with open(pkl_filename, 'rb') as file:
    model_tree = pickle.load(file)

#On l'applique
data_TD['Prediction'] = model_tree.predict(data_TD)
data_TD.to_csv("predictions.txt", columns =['Prediction'], sep = "\t")

#Evaluation
print(1.0 - metrics.accuracy_score(data['V200'],data_TD['Prediction']))#0.006724815981629995

print('Finie en',time.time()-debut_algo,'secondes')