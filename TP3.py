#!/usr/bin/env python3 
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  12 10:44:20 2023

@author: duval
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression

#data = pd.read_csv("https://www.stat4decision.com/telecom.csv")
#
#
#
#data["Churn?"] = data["Churn?"].astype('category')
#
#
#
#
## on définit x et y
#y = data["Churn?"].cat.codes
## on ne prend que les colonnes quantitatives
#x = data.select_dtypes(np.number).drop(["Account Length","Area Code"],axis=1)
#
#y.to_csv('Telecom_y.csv')
#x.to_csv('Telecom_x.csv')

Y, X = pd.read_csv("Telecom_y.csv", index_col = 0) ,  pd.read_csv("Telecom_x.csv", index_col = 0)


X.columns
Y.mean()
X.mean(axis=0)
X.std()

plt.matshow(np.corrcoef(X, rowvar = False))
plt.colorbar()
# Moyennes par groupe
ind = Y['0']==1
X.loc[ind,].mean()
X.loc[-ind,].mean()

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)


# Attention ! Scikit-learn applique une régularisation sur le modèle. 
# Ceci s’explique par l’objectif prédictif du machine learning mais pose des problèmes 
# si vous comparez les résultats avec d'autres logiciels R, SAS….

SK_logit = LogisticRegression(max_iter =2000, penalty='none')#, fit_intercept=False)
# Problème de pénalité
SK_logit.fit(X_train, y_train)



# Sorties: Coeffs, odds ratio

pd.DataFrame(np.concatenate([SK_logit.intercept_.reshape(-1,1),
                             SK_logit.coef_],axis=1),
             index = ["coef"],
             columns = ["constante"]+list(X.columns)).T

params = SK_logit.coef_

print(np.exp(params))

# La première façon de tester la qualité du modèle consiste à regarder la matrice de confusion.
import seaborn as sns
from sklearn.metrics import confusion_matrix

conf = confusion_matrix(y_test, SK_logit.predict(X_test))
sns.heatmap(conf, annot=True)



# Erreur importante sur la classe 1, en raison des données déséquilibrées:


#1 Changement de seuil 
s = 1/4
Y_pred = SK_logit.predict_proba(X_test)[:,1] > s
conf = confusion_matrix(y_test, Y_pred)
sns.heatmap(conf, annot=True)

#2 Sous échantillonnage de la classe 0
# NearMiss vise à séparer les observations correspondantes à des classes différentes. 
# Supprime les observations de la classe majoritaire lorsque des observations associées 
# à des classes différentes sont proches l’une de l’autre.

from imblearn.under_sampling import NearMiss
# Choix de la taille du nouveau dataset 
# Sous-Echantillonnage en utilisant la méthode NearMiss 
nearmiss = NearMiss()
X_under_sample, y_under_sample = nearmiss.fit_resample(X, Y)

# Attention ! Scikit-learn applique une régularisation sur le modèle. 
# Ceci s’explique par l’objectif prédictif du machine learning mais pose des problèmes 
# si vous comparez les résultats avec d'autres logiciels R, SAS….

SK_logit2 = LogisticRegression(max_iter =2000, penalty='none')#, fit_intercept=False)
# Problème de pénalité
SK_logit2.fit(X_under_sample, y_under_sample)

# Sorties: Coeffs, odds ratio

pd.DataFrame(np.concatenate([SK_logit2.intercept_.reshape(-1,1),
                             SK_logit2.coef_],axis=1),
             index = ["coef"],
             columns = ["constante"]+list(X.columns)).T


conf = confusion_matrix(y_test, SK_logit2.predict(X_test))
sns.heatmap(conf, annot=True)


#3 sur echantillonnage
from imblearn.over_sampling import SMOTENC
# Choix de la taille du nouveau dataset 
# Sur-Echantillonnage en utilisant la méthode SMOTE
smote = SMOTENC(categorical_features=[1])
X_over_sample, y_over_sample = smote.fit_resample(X,Y)
y_over_sample.mean()
SK_logit3 = LogisticRegression(max_iter =2000, penalty='none')#, fit_intercept=False)
# Problème de pénalité
SK_logit3.fit(X_over_sample, y_over_sample)

# Sorties: Coeffs, odds ratio

pd.DataFrame(np.concatenate([SK_logit3.intercept_.reshape(-1,1),
                             SK_logit3.coef_],axis=1),
             index = ["coef"],
             columns = ["constante"]+list(X.columns)).T


conf = confusion_matrix(y_test, SK_logit3.predict(X_test))
sns.heatmap(conf, annot=True)


# Pas de sorties stats... On va regarder une autre fonction


import statsmodels.api as sm
# on ajoute une colonne pour la constante
x_stat = sm.add_constant(X_train)
# on ajuste le modèle
model = sm.Logit(y, x_stat)
result = model.fit()
result.summary()


from sklearn.metrics import roc_curve, auc
probas = SK_logit.predict_proba(X_test)
fpr0, tpr0, thresholds0 = roc_curve(y_test, probas[:, 0], pos_label=SK_logit.predict(X_test), drop_intermediate=False)
probas = SK_logit2.predict_proba(X_test)
fpr2, tpr2, thresholds0 = roc_curve(y_test, probas[:, 0], pos_label=SK_logit.predict(X_test), drop_intermediate=False)
probas = SK_logit3.predict_proba(X_test)
fpr3, tpr3, thresholds0 = roc_curve(y_test, probas[:, 0], pos_label=SK_logit.predict(X_test), drop_intermediate=False)


auc_SK = auc(fpr0, tpr0) 
auc_SK2 = auc(fpr2, tpr2) 
auc_SK3 = auc(fpr3, tpr3) 



fig, ax = plt.subplots(1, 1, figsize=(4,4))
ax.plot([0, 1], [0, 1], 'k--')


ax.plot(fpr0, tpr0, label= 'Init auc=%1.5f' % auc_SK)
ax.plot(fpr2, tpr2, label= 'Under auc=%1.5f' % auc_SK2)
ax.plot(fpr3, tpr3, label= 'Over auc=%1.5f' % auc_SK3)
ax.set_title('Courbe ROC')

ax.set_xlabel("FPR")
ax.set_ylabel("TPR");
ax.legend();










