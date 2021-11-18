# -*- coding: utf-8 -*-
"""
Created on Wed Oct 20 14:47:10 2021

@author: USER
"""
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import matthews_corrcoef
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
import statistics
from sklearn import svm
#################################################################################################################################
## PREPROCESAMIENTO
stars_dataframe = pd.read_csv("stars.csv")

################################################
## CARACTERISTICS

#Temperature in Kelvin
Temp = stars_dataframe['Temperature']

#Relative Luminosity
L = stars_dataframe['L']

#Relative Radius
R = stars_dataframe['R']

#Absolute Magnitude
AM = stars_dataframe['A_M']

#General Obs. Color
## Replace string to numbers/ 11 colors
Colors = stars_dataframe['Color']
Color = Colors.str.upper().str.replace("-"," ")
Color = Color.str.replace("WHITE YELLOW","YELLOW WHITE")
All_colors= Color.drop_duplicates().tolist()  ## Se ve que hay 11 
i=0
for x in All_colors:
    Color=Color.replace(x,i)
    i+=1

#SMASS Spec.
## Replace string to numbers/ 7 types
Spec = stars_dataframe['Spectral_Class']
Spec = Spec.str.upper()
All_Spec= Spec.drop_duplicates().tolist()  ## Se ve que hay 7
i=0
for x in All_Spec:
    Spec=Spec.replace(x,i)
    i+=1
###############################################
## LABELS
Tipos = stars_dataframe['Type']
# Red Dwarf - 0
#Brown Dwarf - 1
#White Dwarf - 2
#Main Sequence - 3
#Super Giants - 4
#Hyper Giants - 5
################################################

#Se guardan los datos limpios en csv
stars_dataframe_clean = pd.concat([Temp,L,R,AM,Color,Spec,Tipos],axis=1)
stars_dataframe_clean.to_csv('Stars_clean.csv', header=True, index=False)
print(stars_dataframe_clean)

# SELECCION METODOS DE EVALUACION
X_complete  = stars_dataframe_clean.iloc[:,:-1].values
y_complete  = stars_dataframe_clean.iloc[:,-1].values

# ## Se usaran los primeros 180 datos para entrenar 
# stars_train = stars_dataframe_clean.head(180)
# X_train     = stars_train.iloc[:,:-1].values
# y_train     = stars_train.iloc[:,-1].values

# ## Se usaran los ultimos 60 datos para probar(10 tipo)
# stars_test  = stars_dataframe_clean.tail(60)
# X_test      = stars_test.iloc[:,:-1].values
# y_test      = stars_test.iloc[:,-1].values

X_train, X_test, y_train, y_test = train_test_split(X_complete, y_complete, test_size=0.25, random_state=178)


##################################################################################################################################

## NORMALIZACION
scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test  = scaler.transform(X_test)

##################################################################################################################################

## Representación y reducción dimensional 
pca=PCA(n_components=3)
pca.fit(X_train)
X_pca=pca.transform(X_train)
X_pca_test=pca.transform(X_test)
expl =pca.explained_variance_ratio_
mat_cov= pca.get_covariance()
print('\n')
print("Matriz covarianza",expl)
print('suma:',sum(expl[0:5]))
print('\n')
##Conservare 2 caracteristicas ya que con 2 la suma no disminuye mucho

##################################################################################################################################

## Entrenamiento Knn
k_range = range(1, int(np.sqrt(len(y_train))))
dis=['manhattan','chebyshev', 'minkowski']

MCC=[]
F1=[]
distancia=[]
ki=[]
for i in dis:
    for k in k_range:#Se varia k  
        knn = KNeighborsClassifier(n_neighbors = k,weights='distance',metric=i, metric_params=None,algorithm='brute')
        knn.fit(X_train, y_train)
        y_pred=knn.predict(X_test)
        ## Metricas 
        MCC.append(matthews_corrcoef(y_test,y_pred))
        F1.append(f1_score(y_test,y_pred,average='micro'))
        distancia.append(i)
        ki.append(k)

## Metricas de evaluacion 
print("########################################################################"+"\n")
maximo_MCC = MCC.index(max(MCC))
print("Con knn: Segun MCC({}) el mejor k es {} y la distancia es {}:".format(max(MCC),ki[maximo_MCC],distancia[maximo_MCC]))
maximo_F1  = F1.index(max(F1))
print("Con knn: Segun F1({}) el mejor k es {} y la distancia es {}:".format(max(F1),ki[maximo_F1],distancia[maximo_F1]))
#print(classification_report(y_test, y_pred))

##################################################################################################################################

MCC=[]
F1=[]
distancia=[]
ki=[]
## Entrenamiento con PCA knn 
for i in dis:
    for k in k_range:#Se varia k  
        knn = KNeighborsClassifier(n_neighbors = k,weights='distance',metric=i, metric_params=None,algorithm='brute')
        knn.fit(X_pca, y_train)
        y_pred=knn.predict(X_pca_test)
        ## Metricas 
        MCC.append(matthews_corrcoef(y_test,y_pred))
        F1.append(f1_score(y_test,y_pred,average='micro'))
        distancia.append(i)
        ki.append(k)

## Metricas de evaluacion 
print("\n"+"########################################################################"+"\n")
maximo_MCC = MCC.index(max(MCC))
print("Con knn: Segun MCC({}) y PCA(2 componentes) el mejor k es {} y la distancia es {}:".format(max(MCC),ki[maximo_MCC],distancia[maximo_MCC]))
maximo_F1  = F1.index(max(F1))
print("Con knn: Segun F1({}) y PCA(2 componentes) el mejor k es {} y la distancia es {}:".format(max(F1),ki[maximo_F1],distancia[maximo_F1]))
#print(classification_report(y_test, y_pred))


##################################################################################################################################

## Entrenamiento Logistic regresion
print("\n"+"########################################################################"+"\n")
DC=np.ones((len(X_train),1))
DC_2= np.ones((len(X_test),1))
X_train_lg = np.hstack((X_train,X_train**2,DC))
X_test_lg  = np.hstack((X_test,X_test**2,DC_2))
clf = LogisticRegression(random_state=100, solver='liblinear', max_iter=100000000).fit(X_train_lg,y_train)
y_predict_rg=clf.predict(X_test_lg)
MCC_r=matthews_corrcoef(y_test,y_predict_rg)
F1_r=f1_score(y_test,y_predict_rg,average='micro')
print("Con regresión logistica y una hipotesis X+X^2+1: el MCC es: ", MCC_r)
print("Con regresión logistica y una hipotesis X+X^2+1: el F1 es: ", F1_r)
#print(classification_report(y_test, y_predict))

##################################################################################################################################
## Entrenamiento Logistic regresion y PCA
print("\n"+"########################################################################"+"\n")
DC=np.ones((len(X_pca),1))
DC_2= np.ones((len(X_pca_test),1))
X_train_lg = np.hstack((X_pca,X_pca**2,DC))
X_test_lg  = np.hstack((X_pca_test,X_pca_test**2,DC_2))
clf = LogisticRegression(random_state=100, solver='liblinear', max_iter=100000000).fit(X_train_lg,y_train)
y_predict=clf.predict(X_test_lg)
MCC_r=matthews_corrcoef(y_test,y_predict)
F1_r=f1_score(y_test,y_predict,average='micro')
print("Con regresión logistica, una hipotesis X+X^2+1 y PCA(2 componentes) : el MCC es: ", MCC_r)
print("Con regresión logistica, una hipotesis X+X^2+1 y PCA(2 componentes) : el F1 es: ", F1_r)
#print(classification_report(y_test, y_predict))


##################################################################################################################################
print("\n"+"########################################################################"+"\n")
#print("Segun lo analizado el mejor resultado se vio con regresión logistica sin usar PCA")
#print(classification_report(y_test, y_predict_rg))


##################################################################################################################################

F1=[]
distancia=[]
ki=[]
sc=[]
MCC=[]

for i in dis:
    for k in k_range:#Se varia k  
        knn = KNeighborsClassifier(n_neighbors = k,weights='distance',metric=i, metric_params=None,algorithm='brute')
        scores = cross_val_score(knn, X_complete, y_complete, cv=5,scoring='f1_micro')
        mean = statistics.mean(scores)
        ki.append(k)
        distancia.append(i)
        F1.append(mean)
        #scores = cross_val_score(knn, X_complete, y_complete, cv=5,scoring='mcc')
        ##MCC.append(mean)



maximo_F1  = F1.index(max(F1))
print("Con knn y cross validation: Segun el promedio de F1({}) el mejor k es {} y la distancia es {}:".format(max(F1),ki[maximo_F1],distancia[maximo_F1]))

##################################################################################################################################

F1=[]
ki=[]
sc=[]
MCC=[]


# for k in k_range:#Se varia k  
#     clf = svm.SVC(kernel='linear', C=k, random_state=42)
#     scores = cross_val_score(clf, X_complete, y_complete, cv=5,scoring='f1_micro')
#     mean = statistics.mean(scores)
#     ki.append(k)
#     F1.append(mean)

    
clf = svm.SVC(kernel='linear', C=1, random_state=42)
scores = cross_val_score(clf, X_complete, y_complete, cv=5,scoring='f1_micro')
mean = statistics.mean(scores)
ki.append(1)
F1.append(mean)

maximo_F1  = F1.index(max(F1))
print("Con SVM linear y cross validation: Segun el promedio de F1({}) el mejor C es {}:".format(max(F1),ki[maximo_F1]))




