#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 12 10:35:18 2020

@author: pierre
"""


import pandas
import scipy
from sklearn import metrics
import numpy
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder

from sklearn import model_selection
from sklearn.tree import DecisionTreeClassifier

import pickle
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt
import os
os.chdir('/Volumes/PierreLGZ/Mes documents/M1/DataMining/TD_Note')
from sklearn import metrics

#Fonction permettant d'obtenir les variable ayant une trop grosse correlation
def correlation(dataset, threshold):
    to_drop=[]
    col_corr = set() #contient le nom des variables supprimées
    corr_matrix = dataset.corr()
    for i in range(len(corr_matrix.columns)):
        for j in range(i):
            if (corr_matrix.iloc[i, j] >= threshold) and (corr_matrix.columns[j] not in col_corr):
                if corr_matrix.columns[i] == 'V200':# On supprime les variable trop corrélées a V200 si V200 en a
                    colname = corr_matrix.columns[j] 
                else:
                    colname = corr_matrix.columns[i]
                col_corr.add(colname)
                print(corr_matrix.columns[i],'avec',corr_matrix.columns[j])
                if colname in dataset.columns:
                    to_drop.append(colname)
    print('Supprimer',to_drop, 'de taille',len(to_drop))
    return to_drop

#%%

data = pd.read_csv('data_avec_etiquettes.txt', delimiter = "\t")
V200 = data['V200']
data_tree= data

#On supprime les colonnes ayant une seule modalité
for colname1 in data_tree.columns:
    if len(data_tree[colname1].value_counts()) == 1:
        del data_tree[colname1]
        print('Supprimer',colname1)

cols = data_tree.columns
num_cols = data_tree._get_numeric_data().columns
#On standardise seulement les variables numériques 
standard = StandardScaler()
Data_array = standard.fit_transform(data_tree[num_cols])
data_tree[num_cols] = pd.DataFrame(Data_array)

categ_cols = list(set(cols) - set(num_cols))
#On utilise LabelEncoder pour les varaibles qualitatives
label_encoder = LabelEncoder()
data_tree[categ_cols] = data_tree[categ_cols].apply(label_encoder.fit_transform)

#On supprime les varaibles corrélées à plus de 0.8
to_drop = correlation(data_tree,0.8)
data_tree = data_tree.drop(to_drop, axis=1)

X = data_tree.drop('V200',axis=1)
y = V200

#On sépare la base de donnée en un echantillon train et test pour mieux appliquer le modèle
X_app,X_test,y_app,y_test = model_selection.train_test_split(X,y,test_size = 0.30,random_state=4,stratify = y) 

#On entraine le modèle
tree = DecisionTreeClassifier(min_samples_split=30,min_samples_leaf=10,random_state=5)
tree.fit(X_app, y_app)
#On test le modele sur des donne qu'il n'a jamais vu
print(tree.score(X_test,y_test))#0.9994602144298178

#Pour améliorer le modèle, on supprime les variables inutiles
impVarFirst={"Variable":X.columns,"Importance":tree.feature_importances_} 
b = pd.DataFrame(impVarFirst).sort_values(by="Importance",ascending=False)
col_Tree = b[b['Importance']>0.01]

X_new_app = X_app[col_Tree['Variable']]
X_new_test = X_test[col_Tree['Variable']]

print(list(col_Tree['Variable']))
print(len(list(col_Tree['Variable'])))

#On entraine un nouveau modele sur les nouvelles données
tree = DecisionTreeClassifier(min_samples_split=30,min_samples_leaf=10,random_state=5)
tree.fit(X_new_app, y_app)
#On recupere son score
print(tree.score(X_new_test,y_test))#0.9977936264818801

#Rapport
#seuil de 0.1 : 0.979859250912575 (2)
#seuil de 0.01 : 0.9977396479248618 (4)


#affichage graphique de l'arbre
plt.figure(figsize=(15,10))
plot_tree(tree,feature_names = list(X_new_test.columns),filled=True) 
plt.show()

#Enregistrement du modele
pkl_filename = "modele_1.pkl"
with open(pkl_filename, 'wb') as file:
    pickle.dump(tree, file)

#Evaluation du modele
predFirst = tree.predict(X=X_new_test)
print(metrics.classification_report(predFirst,y_test))
