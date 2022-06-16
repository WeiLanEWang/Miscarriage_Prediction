#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 16 18:05:00 2022

@author: erica
"""


import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SVMSMOTE
from imblearn.pipeline import Pipeline
import seaborn as sns
from collinearity import SelectNonCollinear
from imblearn.combine import  SMOTEENN
from collections import Counter

#define seed for reproduction
seed = 7

#Balancing numbers in both centres by randomized under sampling
df = pd.read_csv('both centres.csv')
feature = df.drop('centre', axis = 1)
centre = df['centre']

rus = RandomUnderSampler(random_state=seed)
feature_balanced, centre_balanced = rus.fit_resample(feature, centre)
Counter(centre_balanced)

#drop some features with >50% missing data and other irrelavant data
feature_balanced.isna().sum()

todrop = ['HR', 'DIABP', 'SYSBP',  'BMI']

feature_balanced.drop(labels=todrop, axis=1, inplace=True)

#seperate the patients from IVF centre and fill NAs by mean value
df = pd.concat([feature_balanced, centre_balanced], axis=1)
X = df.loc[df['centre']=='IVF'].drop(['pregnancy', 'centre'], axis = 1)
y = df.loc[df['centre']=='IVF']['pregnancy']
X.isna().sum()
X.fillna(X.mean(),inplace=True)
X_names = list(X.columns)

#seperate the patients from personal centre and fill NAs by mean value
X2 = df.loc[df['centre']=='personal'].drop(['pregnancy', 'centre'], axis = 1)
y2 = df.loc[df['centre']=='personal'].iloc[888:]['pregnancy']
X2.fillna(X2.mean(),inplace=True)
X_names2 = list(X2.columns)

#Standard Scaler
scaler = StandardScaler()
X = scaler.fit_transform(X)
X = pd.DataFrame(X, columns=X_names)
X2 = scaler.fit_transform(X2)
X2 = pd.DataFrame(X2, columns=X_names2)

#removing collinearity features
X = X.to_numpy()
selector = SelectNonCollinear(0.3)
selector.fit(X,y)
mask = selector.get_support()
X = pd.DataFrame(X[:,mask],columns = np.array(X_names)[mask])
sns.heatmap(X.corr().abs(),annot=True)
list(X.columns)

X2 = X2.to_numpy()
selector = SelectNonCollinear(0.3)
selector.fit(X2,y2)
mask= selector.get_support()
X2 = pd.DataFrame(X2[:,mask],columns = np.array(X_names2)[mask])
sns.heatmap(X2.corr().abs(),annot=True)
list(X2.columns)


#Balance pregnancy outcomes by SVM SMOTE and randomized under sampling 
smote = SMOTEENN(random_state=seed)
X_smote, y_smote = smote.fit_resample(X, y)

X_smote2, y_smote2 = smote.fit_resample(X2, y2)

#Random undersampling
X_rus, y_rus = rus.fit_resample(X, y)
Counter(y_rus)

#SMOTE and RUS pipeline
over = SVMSMOTE(sampling_strategy=0.5)
under = RandomUnderSampler(sampling_strategy=1)
steps = [('o', over), ('u', under)]
pipeline = Pipeline(steps=steps)

X_pip, y_pip = pipeline.fit_resample(X, y)

X_pip2, y_pip2 = pipeline.fit_resample(X2, y2)

#removing collinearity features
X = X.to_numpy()
selector = SelectNonCollinear(0.3)
selector.fit(X,y)
mask = selector.get_support()
X_noncol = pd.DataFrame(X[:,mask],columns = np.array(X_names)[mask])
sns.heatmap(X_noncol.corr().abs(),annot=True)
list(X_noncol.columns)
