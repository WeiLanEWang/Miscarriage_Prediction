#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 16 18:26:18 2022

@author: erica
"""

import numpy as np
import pandas as pd

import tqdm as tqdm
from imblearn.over_sampling import SVMSMOTE
from imblearn.combine import SMOTEENN
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
 
from collections import Counter
from pprint import pprint

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data import random_split
from torch import sigmoid
from torch.utils.data import Subset
from torch.autograd import Variable
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.metrics import roc_auc_score, classification_report
from sklearn.preprocessing import OneHotEncoder

from captum.attr import (
    GradientShap,
    DeepLift,
    IntegratedGradients)

#%%
seed = 7
SEED = 5

'''prepare train and test sets'''

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,
                                                    random_state = seed)
X_train, y_train = smote.fit_resample(X_train, y_train)
Counter(y_train)
Counter(y_test)


X_train_tr = torch.tensor(X_train.to_numpy(), dtype=torch.float)
X_test_tr = torch.tensor(X_test.to_numpy(), dtype=torch.float)
y_train_tr = torch.tensor(y_train.to_numpy(), dtype=torch.long)
y_test_tr = torch.tensor(y_test.to_numpy(), dtype=torch.long)


#make tables for model performance and interpretable outcomes

tuning_report = pd.DataFrame(columns=['dropout rate','learning rate',
                                      'accuracy', 'precision',
                                   'recall', 'F1 score', 'test AUC ROC'])

#define a base network model
EPOCHS = 500

hidden_layer1 = 32
hidden_layer2 = 64
hidden_layer3 = 32


dropout=0.5
class Net(nn.Module):
    def __init__(self, input_dim):
        super(Net, self).__init__()
        self.layer1 = nn.Linear(input_dim, hidden_layer1)
        self.layer2 = nn.Linear(hidden_layer1, hidden_layer2)
        self.layer3 = nn.Linear(hidden_layer2, hidden_layer3)
        self.layer4 = nn.Linear(hidden_layer3, 2)
        
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = self.dropout(x)
        x = F.relu(self.layer2(x))
        x = self.dropout(x)
        x = F.relu(self.layer3(x))
        x= F.sigmoid(self.layer4(x))
        return x

#DNN tuning and records
learn_rate=0.0005
weight_decay=0.001

model = Net(X_train.shape[1])
optimizer = torch.optim.Adam(model.parameters(), lr=learn_rate, weight_decay=weight_decay)
loss_fn = nn.CrossEntropyLoss()
model


loss_list = np.zeros((EPOCHS,))
test_loss_list = np.zeros((EPOCHS,))
accuracy_list = np.zeros((EPOCHS,))

for epoch in tqdm.trange(EPOCHS):
    y_pred = model(X_train_tr)
    loss = loss_fn(y_pred, y_train_tr)
    y_pred_test = model(X_test_tr)
    test_loss = loss_fn(y_pred_test, y_test_tr)
    loss_list[epoch] = loss.item()
    test_loss_list[epoch] = test_loss.item()
    
    correct = (torch.argmax(y_pred, dim=1) == y_train_tr).type(torch.FloatTensor)
    accuracy_list[epoch] = correct.mean()

    #zero gradients
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if epoch % 20 == 0:
        print ('Epoch {}/{} => Train Loss: {:.2f}'.format(epoch+1, EPOCHS, loss.item()))
        print ('Epoch {}/{} => Test Loss: {:.2f}'.format(epoch+1, EPOCHS, test_loss.item()))

#DNN performance
enc = OneHotEncoder()

Y_onehot = enc.fit_transform(y_test_tr[:, np.newaxis]).toarray()

with torch.no_grad():
    y_pred = model(X_test_tr).numpy()
    fpr, tpr, threshold = roc_curve(Y_onehot.ravel(), y_pred.ravel())
out_probs_test = model(X_test_tr).detach().numpy()
out_classes_test = np.argmax(out_probs_test, axis=1)
    
train_probs = model(X_train_tr).detach().numpy()
test_probs = model(X_test_tr).detach().numpy()
test_prediction = np.argmax(test_probs, axis=1)

new_row = {'dropout rate': dropout,
           'learning rate': learn_rate, 
           'accuracy': sum(out_classes_test == y_test) / len(y_test), 
           'precision': precision_score(test_prediction, y_test),
           'recall':recall_score(test_prediction, y_test), 
           'F1 score':f1_score(test_prediction, y_test), 
           'test AUC ROC': auc(fpr, tpr)}

tuning_report =tuning_report.append(new_row, ignore_index=True)

#training curve
accuracy_list.mean()
plt.plot(fpr, tpr, label='AUC = {:.3f}'.format(auc(fpr, tpr)))
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('ROC curve')
plt.legend();


#training and test accuracy
out_probs_train = model(X_train_tr).detach().numpy()
out_classes_train = np.argmax(out_probs_train, axis=1)
print("Train Accuracy:", sum(out_classes_train == y_train) / len(y_train))

out_probs_test = model(X_test_tr).detach().numpy()
out_classes_test = np.argmax(out_probs_test, axis=1)
print("Test Accuracy:", sum(out_classes_test == y_test) / len(y_test))

# One hot encoding
enc = OneHotEncoder()
Y_onehot = enc.fit_transform(y_test_tr[:, np.newaxis]).toarray()

with torch.no_grad():
    y_pred = model(X_test_tr).numpy()
    fpr, tpr, threshold = roc_curve(Y_onehot.ravel(), y_pred.ravel())

print(auc(fpr, tpr))



'''inpterpretable analysis'''

'''Integrated Gradients'''

ig = IntegratedGradients(model)

X_test_tr.requires_grad_()
attr, delta = ig.attribute(X_test_tr,target=0, return_convergence_delta=True)
attr = attr.detach().numpy()

# Helper method to print importances and visualize distribution
def visualize_importances(X_names, importances, title="Average Feature Importances", plot=True, axis_title="Features"):
    print(title)
    for i in range(len(X_names)):
        print(X_names[i], ": ", '%.3f'%(importances[i]))
    x_pos = (np.arange(len(X_names)))
    
    if plot:
        plt.figure(figsize=(12,6))
        plt.bar(x_pos, importances, align='center')
        plt.xticks(x_pos, X_names, wrap=True)
        plt.xlabel(axis_title)
        plt.title(title)
        
visualize_importances(X_names, np.mean(attr, axis=0))

attr_mean = np.mean(attr,axis=0).tolist()

ig_importance = attr.mean(axis=0)
name = np.array(X_names)

ig_importance = pd.DataFrame(np.c_[name, ig_importance], columns=['Featre', 'Importance'])


ig_importance.reindex(ig_importance.Importance.abs().sort_values().index)

'''Gradient Shap'''
gs = GradientShap(model)
baseline_dist = torch.zeros(X_train.shape[0], X_train.shape[1]) 
attr_gs, delta = gs.attribute(X_train_tr, stdevs=0.09, n_samples=4, 
                                   baselines=baseline_dist,
                                   target=1, return_convergence_delta=True)
attr_gs = attr_gs.detach().numpy()
visualize_importances(X_names, np.mean(attr_gs, axis=0))


gs_importance = attr_gs.mean(axis=0)
name = np.array(X_noncol.columns)

gs_importance = pd.DataFrame(np.c_[name, gs_importance], columns=['Featre', 'Importance'])

'''DeepLIFT'''
dl = DeepLift(model)
attr_dl = dl.attribute(X_train_tr, baseline_dist, target=1, return_convergence_delta=False)

attr_dl = attr_dl.detach().numpy()

visualize_importances(X_names, np.mean(attr_dl, axis=0))


dl_importance = attr.mean(axis=0)
name = np.array(X_names)

dl_importance = pd.DataFrame(np.c_[name, dl_importance], columns=['Featre', 'Importance'])


