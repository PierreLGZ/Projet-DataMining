#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 18 11:04:19 2020

@author: pierre
"""

import os
os.chdir('/Volumes/PierreLGZ/Mes documents/M1/DataMining/TD_Note')
from collections import Counter
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from scipy.cluster.hierarchy import dendrogram,linkage
import matplotlib.pyplot as plt

from scipy.cluster.hierarchy import fcluster

from sklearn import model_selection
from sklearn.tree import DecisionTreeClassifier
from sklearn.decomposition import PCA
from sklearn.tree import plot_tree
from sklearn import metrics
import pickle
from sklearn.preprocessing import LabelEncoder

#Fonction permettant d'afficher la courbe de gain
def correlation(dataset, threshold):
    to_drop=[]
    col_corr = set() # Set of all the names of deleted columns
    corr_matrix = dataset.corr()
    for i in range(len(corr_matrix.columns)):
        for j in range(i):
            if (corr_matrix.iloc[i, j] >= threshold) and (corr_matrix.columns[j] not in col_corr):
                if corr_matrix.columns[i] == 'V200':
                    colname = corr_matrix.columns[j] # getting the name of column
                else:
                    colname = corr_matrix.columns[i]
                col_corr.add(colname)
                print(corr_matrix.columns[i],'avec',corr_matrix.columns[j])
                if colname in dataset.columns:
                    to_drop.append(colname)
    print('Supprimer',to_drop, 'de taille',len(to_drop))
    return to_drop

#Fonction pour obtenir la somme des carré de la distance en fonction du nombre de cluster
def find_optimal_clusters(data, max_k):
    iters = range(2, max_k+1, 1)
    sse = []
    s = 0
    pourc = []
    av=0
    for k in iters:
        fit = KMeans(n_clusters=k,random_state=20).fit(data)
        sse.append(fit.inertia_)
        s+= fit.inertia_
    print(s)
    for i in sse:
        
        pourc.append(av+i/s)
        av = av + i/s
    print(pourc)
        
    f, ax = plt.subplots(1, 1)
    ax.plot(iters, pourc, marker='o')
    ax.set_xlabel('Cluster Centers')
    ax.set_xticks(iters)
    ax.set_xticklabels(iters)
    ax.set_ylabel('SSE')
    ax.set_title('SSE by Cluster Center Plot')
    ax.axvline(x=11, color='green', linestyle='-')
    ax.axvline(x=3, color='orange', linestyle='-')
    ax.axvline(x=14, color='purple', linestyle='-')
    
#%%
data = pd.read_csv('data_avec_etiquettes.txt', delimiter = "\t")
V200 = data['V200']
classes = data['V200'].unique()

#On supprime les colonnes ayant une seule modalité
for colname1 in data.columns:
    if len(data[colname1].value_counts()) == 1:
        del data[colname1]
        print('Supprimer',colname1)

#Ici, on recupere juste les colonne numeric pour plus tard sans standardiser 
#car on aimerait regrouper les données via leur moyenne.
#i on standardise maintenant,on obtiendrait juste une moyenne de 0.
cols = data.columns
num_cols = data._get_numeric_data().columns

categ_cols = list(set(cols) - set(num_cols))
#On utilise LabelEncoder pour les varaibles qualitatives
label_encoder = LabelEncoder()
data[categ_cols] = data[categ_cols].apply(label_encoder.fit_transform)

#On supprime les varaibles corrélées à plus de 0.8
to_drop = correlation(data,0.8)
data = data.drop(to_drop, axis=1)

#On recupere le barycentre de chaque modalité V200
barycentres = {}
for i in classes :
    barycentres[i] = np.mean(data[V200==i].drop('V200',axis=1))
    print(i)

#On crée un nouveau dataset a partir des barycentres
columns = range(len(barycentres['m12']))
df = pd.DataFrame(columns=columns)
n = 0
for i in classes:
    df.loc[n] = np.array(barycentres[i])
    n+=1

df['Classe'] = classes

#On peut désormais standardiser les données
standard = StandardScaler()
Data_array = standard.fit_transform(df.iloc[:,:-1])
num = pd.DataFrame(Data_array)


#On fait une PCA pour ne prendre que les dimensions avec le plus de variabilité
mypca = PCA(n_components=13) # On paramètre ici pour ne garder que 13 composantes
mypca.fit(num)

print(mypca.explained_variance_ratio_)  
print(mypca.singular_values_)
print(mypca.components_)

#On affiche le diagramme de pareto
y = list(mypca.explained_variance_ratio_)
x = range(len(y))
ycum = np.cumsum(y)
plt.bar(x,y)
plt.plot(x,ycum,"-r")
plt.show()

#On transforme nos données en fonction de nos 13 composantes
data_sortie = mypca.fit_transform(num)
pd.DataFrame(data_sortie)

#On peut donc commencer le clsutering via la CAH
Z = linkage(data_sortie,'ward')

#On affiche la CAH - la ligne coupe 11 fois les branches, on a donc 11 clusters
plt.title('CAH avec matérialisation des classes') 
dendrogram(Z,labels=classes,color_threshold=17) 
plt.axhline(y=17, color='green', linestyle='-')
plt.show()

#On valide avec la méthode du coude via KMeans
find_optimal_clusters(data_sortie,20)

#On recupere les clusters
groupes_cah = fcluster(Z,t=17,criterion='distance') 

cah = pd.DataFrame(groupes_cah,classes).reset_index()
cah.columns = ['V200','Cluster']

#Pour mieux visualiser chaque cluster
occurence_V200 = Counter(classes)
cluster = cah.groupby('Cluster')['V200'].apply(list).reset_index(name='new')

#On crée un dataframe nous indiquant pour chaque modalité, son cluster
dendro = pd.DataFrame({'V200_Prim' : groupes_cah,'V200': classes})
#dendro.to_csv("clustering.csv")

#On regroupe sur toute les données pour pouvoir faire la classificaiton ensuite
data['V200']=V200
df_dendro = pd.merge(data,dendro,on="V200")

#%%
d = df_dendro.drop(['V200','V200_Prim'],axis=1)

cols = d.columns
num_cols = d._get_numeric_data().columns
#On restandardise les données 
standard = StandardScaler()
Data_array = standard.fit_transform(d[num_cols])
d[num_cols]=pd.DataFrame(Data_array)
#La transformation des variables qualitative a déjà été faite auparavant

X = d
y= df_dendro['V200_Prim']

#On sépare la base de donnée en un echantillon train et test pour mieux appliquer le modèle
X_app,X_test,y_app,y_test = model_selection.train_test_split(X,y,test_size = 0.30,random_state=5,stratify = y) 

#On entraine le modèle
tree = DecisionTreeClassifier(min_samples_split=30,min_samples_leaf=10,random_state=5)
tree.fit(X_app, y_app)
#On test le modèle sur des données qu'il n'a jamais vu
print(tree.score(X_test,y_test))#0.9998650536074545

#On ne prend que les variables considéré par le modèle comme étant importantes
impVarFirst={"Variable":X.columns,"Importance":tree.feature_importances_} 
b = pd.DataFrame(impVarFirst).sort_values(by="Importance",ascending=False)
col_Tree = b[b['Importance']>0.1]
X_new_app = X_app[col_Tree['Variable']]
X_new_test = X_test[col_Tree['Variable']]
print(list(col_Tree['Variable']))
print(len(list(col_Tree['Variable'])))

#On entraine un nouveau modele avec les nouvelles variables
tree = DecisionTreeClassifier(min_samples_split=30,min_samples_leaf=10,random_state=5)
tree.fit(X_new_app, y_app)
#On le test
print(tree.score(X_new_test,y_test))#0.9998650536074545
#['V165', 'V164']

#affichage graphique de l'arbre
plt.figure(figsize=(15,10))
plot_tree(tree,feature_names = list(X_new_test.columns),filled=True) 
plt.show()

#On enregistre le modèle
pkl_filename = "modele_3.pkl"
with open(pkl_filename, 'wb') as file:
    pickle.dump(tree, file)
    
#Evaluation
predFirst = tree.predict(X=X_new_test)
print(metrics.classification_report(predFirst,y_test))