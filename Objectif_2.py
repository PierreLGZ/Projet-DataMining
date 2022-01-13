#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 13 09:08:37 2020

@author: pierre
"""



import os
os.chdir('/Volumes/PierreLGZ/Mes documents/M1/DataMining/TD_Note')

import numpy
import scipy
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder

from sklearn import model_selection
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn import metrics
import pickle

#Fonction pour récupéré les colonnes les plus importantes du jeu de données
def lda_colum(X_app, y_app):
    lda = LinearDiscriminantAnalysis(solver = 'eigen',store_covariance=True)
    lda.fit(X_app, y_app)
    
    p = X_app.shape[1]
    n = X_app.shape[0]
    K = len(y_app.value_counts())
    
    T = numpy.cov(X_app.values,rowvar=False)
    LW = numpy.linalg.det(lda.covariance_)/numpy.linalg.det((n-1)/n*T)
    
    #degrés de liberté - numérateur
    ddlSuppNum = K-1
    #degré de liberté dénominateur
    ddlSuppDenom = n - K - p + 1 #FTest - préparation
    FTest = numpy.zeros(p) #p-value pour FTest
    pvalueFTest = numpy.zeros(p)
    #Tester chaque variable
    for j in range(p): #matricesintermédiaires numérateur
        tempNum = lda.covariance_.copy()
        #supprimer la référence de la variable à traiter
        tempNum = numpy.delete(tempNum,j,axis=0)
        tempNum = numpy.delete(tempNum,j,axis=1)
        #même chose pour dénominateur
        tempDenom = (n-1)/n*T
        tempDenom = numpy.delete(tempDenom,j,axis=0)
        tempDenom = numpy.delete(tempDenom,j,axis=1)
        #lambda sans la variable
        LWVar = numpy.linalg.det(tempNum)/numpy.linalg.det(tempDenom) #print(LWVar)
        #FValue
        FValue = ddlSuppDenom/ddlSuppNum*(LWVar/LW-1)
        #récupération des valeurs
        FTest[j] = FValue
        pvalueFTest[j] = 1 - scipy.stats.f.cdf(FValue,ddlSuppNum,ddlSuppDenom)
    #affichage
    temp = {'var':X_app.columns,'F':FTest,'pvalue':pvalueFTest} 
    d = pd.DataFrame(temp)
    return d 

#Fonction permettant d'afficher la courbe de gain
def scoring(LDA,y_test,X_test,X_new_test_LDA):
    
    proba = LDA.predict_proba(X_new_test_LDA)
    score = proba[:,1]
    pos = pd.get_dummies(y_test).values
    pos = pos[:,1]
    
    npos = numpy.sum(pos)
    
    #index pour tri selon le score croissant
    index = numpy.argsort(score)
    index = index[::-1] #inverse
    sort_pos = pos[index] # [ 1 1 1 1 1 0 1 1 ...]
    
    #mettre les deux meme taille + proportion
    cpos = numpy.cumsum(sort_pos) 
    rappel = cpos/npos 
    n = y_test.shape[0]
    n 
    taille = numpy.arange(start=1,stop=n+1,step=1) #
    taille = taille / n 
    
    #titre et en-têtes
    plt.title('Courbe de gain') 
    plt.xlabel('Taille de cible') 
    plt.ylabel('Rappel')
    #limites en abscisse et ordonnée
    plt.xlim(0,1) 
    plt.ylim(0,1)
    #astuce pour tracer la diagonale
    plt.scatter(taille,taille,marker='.',color='blue')
    plt.scatter(taille,rappel,marker='.',color='red')#insertion du couple (taille, rappel)
    #affichage
    plt.show()

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
data_lda = pd.read_csv('data_avec_etiquettes.txt', delimiter = "\t")

#On supprime les colonnes ayant une seule modalité
for colname1 in data_lda.columns:
    if len(data_lda[colname1].value_counts()) == 1:
        del data_lda[colname1]
        print('Supprimer',colname1)

cols = data_lda.columns
num_cols = data_lda._get_numeric_data().columns
#On standardise seulement les variables numériques 
standard = StandardScaler()
Data_array = standard.fit_transform(data_lda[num_cols])
data_lda[num_cols] = pd.DataFrame(Data_array)

categ_cols = list(set(cols) - set(num_cols))
#On supprime V200 des var qualitative pour ne pas le transformer via LabelEncoder et pouvoir le changer plus tard
for i in range(len(categ_cols)):
    if categ_cols[i] == 'V200':
        categ_cols.pop(i) 
        break
#On utilise LabelEncoder pour les varaibles qualitatives
label_encoder = LabelEncoder()
data_lda[categ_cols] = data_lda[categ_cols].apply(label_encoder.fit_transform)

#On crée une nouvel variables en transformant V200, si présence de m16 alors =1, sinon = 0
data_lda.loc[data_lda['V200'] != 'm16', 'V200_m16'] = 0
data_lda.loc[data_lda['V200'] == 'm16', 'V200_m16'] = 1

#On supprime les varaibles corrélées à plus de 0.8
to_drop = correlation(data_lda.drop('V200', axis=1),0.8)

X = data_lda.drop(['V200', 'V200_m16'], axis=1).drop(to_drop,axis=1)
y = data_lda['V200_m16']

#On sépare la base de donnée en un echantillon train et test pour mieux appliquer le modèle
X_app,X_test,y_app,y_test = model_selection.train_test_split(X,y,test_size = 0.30,random_state=5,stratify = y) 

#On entraine le modèle
lda_avant = LinearDiscriminantAnalysis(solver="eigen")
lda_avant.fit(X_app, y_app)
#On test le modèle sur des données qu'il n'a jamais vu
print(lda_avant.score(X_test,y_test))#0.9985830628782716

#On ne prend que les variables considéré par le modèle comme étant importantes
Importance_LDA = lda_colum(X_app,y_app)
Col_lda = Importance_LDA[Importance_LDA['pvalue']<0.00001].reset_index(drop = True).sort_values('pvalue')
X_new_app = X_app[Col_lda['var']]
X_new_test = X_test[Col_lda['var']]
print(list(Col_lda['var']))
print(len(list(Col_lda['var'])))

#On entraine un nouveau modele avec les nouvelles variables
lda_apres = LinearDiscriminantAnalysis()
lda_apres.fit(X_new_app,y_app)
#On le test
print(lda_apres.score(X_new_test,y_test))#0.9985898101978988

#On affiche la courbe de gain
scoring(lda_apres,y_test,X_test,X_new_test)

#On enregistre le modèle
pkl_filename = "modele_2.pkl"
with open(pkl_filename, 'wb') as file:
    pickle.dump(lda_apres, file)

#Evaluation
predFirst = lda_apres.predict(X=X_new_test)
print(metrics.confusion_matrix(predFirst,y_test))
print(metrics.classification_report(predFirst,y_test))
