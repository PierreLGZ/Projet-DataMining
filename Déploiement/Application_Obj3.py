#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 27 14:54:21 2020

@author: pierre
"""
import time
debut_algo = time.time()
from sklearn.preprocessing import StandardScaler
import pandas as pd
import pickle
from sklearn.preprocessing import LabelEncoder
import os
from sklearn import metrics
os.chdir('/Volumes/PierreLGZ/Mes documents/M1/DataMining/TD_Note')

col = ['V165', 'V164']
data_TD = pd.read_csv('data_avec_etiquettes.txt', delimiter = "\t", usecols = col)

cluster = pd.read_csv('clustering.csv', delimiter = ",",usecols=['V200_Prim','V200'])
classes = pd.read_csv('classes.txt', delimiter = "\t")

#On associe chaque modalité à son cluster
regroupement = pd.merge(classes,cluster,on="V200")

#On standardise les données quantitave
standard = StandardScaler()
data_TD = pd.DataFrame(standard.fit_transform(data_TD))
#Pas de LabelEncoder car pas de varaibles qualitatives

#On recupere le model
pkl_filename = "modele_classif_cluster.pkl"
with open(pkl_filename, 'rb') as file:
    model_tree = pickle.load(file)

#On l'applique
sorties = pd.DataFrame()
sorties['Prediction'] = model_tree.predict(data_TD)
sorties['V200_Prim'] = regroupement['V200_Prim']
sorties.to_csv("sorties.txt", sep = "\t")

#Evaluation
print(metrics.classification_report(sorties['Prediction'],sorties['V200_Prim']))
print('Finie en',time.time()-debut_algo,'secondes')

