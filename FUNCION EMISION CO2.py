#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 24 09:50:17 2021

@author: nestor
"""
import pandas as pd
import numpy as np 
import seaborn as sns
import matplotlib.pyplot as plt 
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import DBSCAN 
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import NearestNeighbors
#%%
consumo = 'FUELCONSUMPTION_HWY'
co2 = 'CO2EMISSIONS'
#%%

df = pd.read_csv('/home/nestor/entornos/neoland/2 Practica/9 CLUSTER/coches.csv')



#%%

#Transformo para sacar cluster con DBSCAN

x = df.loc[:,['FUELCONSUMPTION_HWY', 'CO2EMISSIONS']]
y = df.loc[:,'CO2EMISSIONS']

x = MinMaxScaler().fit_transform(x)

epsilon = 0.05
sample = 4

db = DBSCAN(eps = epsilon,
            min_samples=sample).fit(x)

labels = db.labels_

df['cluster'] = labels
# Ahora junto las clases que me ha separado de más

df.replace({2:1, 3:1, 4:1, 5:1}, inplace = True)

df = df.loc[df['cluster'] != -1 ,:]

#%%

### REGRESION LINEAL 1 ###
def co2_0(patron):

    clust0 = df.loc[df['cluster'] == 0,:]
    
    x = clust0[['FUELCONSUMPTION_HWY']].values
    y = clust0['CO2EMISSIONS'].values
    
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size= 0.3)
    
    # Algoritmo de regresion
    
    regresion = linear_model.LinearRegression()
    regresion.fit(x_train, y_train)
    
    emision = regresion.predict(patron[0][-3].reshape(-1,1))
    
    return emision

#%%

### REGRESION LINEAL 2 ###

clust 1 = df.loc[df['cluster']==1,:]

x1 = clust1[consumo].values
y1 = clust1[co2]

x_train1, x_test1, y_train1, y_test1 = train_test_split(x1, 
                                                        y1, 
                                                        test_size=0.3)

regegresion1 = linear_model.LinearRegression()
regresion1.fit(x_train1, y_train1)


#%%


### CLASIFICADOR ###

df2 = df.drop([consumo, co2, 'MODELYEAR', 'MODEL'])

# convierto a numeros las cateogricas

for i in ('MAKE', 'VEHICLECLASS', 'TRNASMISSION', 'FUELTYPE'):
    llaves = df2[i].unique
    valores = range(len(llaves))
    dic = dict(zip(llaves,valores))
    df2[i].replace(dic, inplace = True)
    
x = df2.values
y = df['cluster'].values

madmax = MinMaxScaler()
madmax.fit(x)
x = madmax.transform(x)

# hago el random

skf = StratifiedKFold(n_splits = 5)

for train_indice, test_indice in skf.split(x,y):
    x_train, x_test = x[train_indice], x[test_indice]
    y_train, y_test = y[train_indice], y[train_indice]
    
    clf = RandomForestClassifier()
    clf.fit(x_train, y_train)
    
#%%
def funcion_supreme(patron):
    
    #Abro archivo
    
    df = pd.read_csv('/home/nestor/entornos/neoland/2 Practica/9 CLUSTER/coches.csv')
    
    #### HAGO DBSCAN ####
    
    x = df.loc[:,['FUELCONSUMPTION_HWY', 'CO2EMISSIONS']]
    y = df.loc[:,'CO2EMISSIONS']
    
    x = MinMaxScaler().fit_transform(x)
    
    epsilon = 0.05
    sample = 4
    
    db = DBSCAN(eps = epsilon,
                min_samples=sample).fit(x)
    
    labels = db.labels_
    
    df['cluster'] = labels
    # Ahora junto las clases que me ha separado de más
    
    df.replace({2:1, 3:1, 4:1, 5:1}, inplace = True)
    
    df = df.loc[df['cluster'] != -1 ,:]
    
    
    
    # Data frame para clasificacion
    df2 = df.drop([consumo, co2, 'MODELYEAR', 'MODEL'], axis = 1)

    # convierto a numeros las cateogricas
    
    for i in ('MAKE', 'VEHICLECLASS', 'TRANSMISSION', 'FUELTYPE'):
        llaves = df2[i].unique()
        valores = range(len(llaves))
        dic = dict(zip(llaves,valores))
        df2[i].replace(dic, inplace = True)
        
    x = df2.values
    y = df['cluster'].values
    
    madmax = MinMaxScaler()
    madmax.fit(x)
    x = madmax.transform(x)
    
    #### CLASIFICADOR ####
    
    skf = StratifiedKFold(n_splits = 5)
    
    for train_indice, test_indice in skf.split(x,y):
        x_train, x_test = x[train_indice], x[test_indice]
        y_train, y_test = y[train_indice], y[train_indice]
        
        clf = RandomForestClassifier()
        clf.fit(x_train, y_train)
    
    patron_transformado = madmax.transform(patron)
    clasif = clf.predict(patron_transformado)
    
    if clasif == np.array([0]):
    
        #### REGRESION LINEL CLUSTER 0 ####
        
        clust0 = df.loc[df['cluster'] == 0,:]
        
        x = clust0[['FUELCONSUMPTION_HWY']].values
        y = clust0['CO2EMISSIONS'].values
        
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size= 0.3)
        
        # Algoritmo de regresion
        
        regresion = linear_model.LinearRegression()
        regresion.fit(x_train, y_train)
        
        emision = regresion.predict(patron[0][-3].reshape(-1,1))
    
    else:
        
        #### REGRESION LINEAL CLUSTER 1 ####
        
        clust1 = df.loc[df['cluster']==1,:]

        x1 = clust1[consumo].values
        y1 = clust1[co2]
        
        x_train1, x_test1, y_train1, y_test1 = train_test_split(x1, 
                                                                y1, 
                                                                test_size=0.3)
        
        regresion1 = linear_model.LinearRegression()
        regresion1.fit(x_train1, y_train1)

        emision = regresion1.predict(patron[0][-3].reshape(-1,1))
        
    return emision



#%%

#### Prueba #####


print(funcion_supreme(np.array([[18. ,  6. ,  1.6,  1. ,  1. ,  2. , 11.1,  8.2,  9.8, 29. ]])))












