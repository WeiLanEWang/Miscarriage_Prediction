#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 16 18:17:02 2022

@author: erica
"""

rom sklearn.model_selection import train_test_split
from collections import Counter
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.metrics import roc_auc_score, classification_report
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV


from pprint import pprint
from sklearn.utils.class_weight import compute_class_weight

from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from interpret.glassbox import ExplainableBoostingClassifier
from sklearn.ensemble import VotingClassifier
from interpret import show

seed = 7
SEED = 5


'''Defined functions'''

#classweight calculator

def classweight_cal(y_train):
    classweights = compute_class_weight(class_weight = 'balanced', 
                                      classes= np.unique(y_train), 
                                      y= y_train)
    classweight = dict(zip(np.unique(y_train), classweights))
    return classweight

#define a function to list most important factors
def imp_df(column_names, importances):
    data = {
        'Feature': column_names,
        'Importance': importances,
    }
    df = pd.DataFrame(data) \
        .set_index('Feature') \
        .sort_values('Importance', ascending=False)

    return df

#define a function for outcome assessment
def assess(model, X_train, y_train, X_test, y_test):
    test_prediction = model.predict(X_test)
    train_probs = model.predict_proba(X_train)[:,1]
    test_probs = model.predict_proba(X_test)[:,1]
    print('Training dataset shape', Counter(y_train))
    print('Accuracy:', accuracy_score(test_prediction, y_test))
    print(f'Train ROC AUC Score: {roc_auc_score(y_train, train_probs)}')
    print(f'Test ROC AUC  Score: {roc_auc_score(y_test, test_probs)}')
    print(classification_report(test_prediction, y_test, digits = 3))
    print('Parameters currently in use:\n')
    pprint(model.get_params())
    

#define a function for outcome assessment
def report_assess(report1, report2): 
    repostd=report1.std()
    repomean = report1.mean()
    repo_IVF = pd.concat([repomean.round(3).rename('mean'), 
                          repostd.round(3).rename('SD')], axis=1)
    repostd2=report2.std()
    repomean2 = report2.mean()
    repo_personal = pd.concat([repomean2.round(3).rename('mean'), 
                          repostd2.round(3).rename('SD')], axis=1)
    
    print('IVF:\n')
    pprint(repo_IVF)
    print('personal:\n')
    pprint(repo_personal)
  
   
report = np.zeros((SEED,6))
report_names = ['accuracy', 'precision','recall', 'F1_score', 'train_ROC', 'test_ROC']

imp_feature = np.zeros((SEED,14))


#%%
'''Random forest model optimization'''

# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 300, stop = 1000, num = 10)]
# Number of features to consider at every split
max_features = ['auto', 'sqrt']
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(1, 25, num = 5)]
max_depth.append(None)
# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10]
# Minimum number of samples required at each leaf node
min_samples_leaf = [2, 4]
# Method of selecting samples for training each tree
bootstrap = [True, False]

#Create the random grid
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}

#First create the base model to tune
report = np.zeros((SEED,6))
report_names = ['accuracy', 'precision','recall', 'F1_score', 'train_ROC', 'test_ROC']
report_rf = pd.DataFrame(report,columns=report_names)
imp_feature = np.zeros((0,X.shape[1]))
imp_feature_rf = pd.DataFrame(imp_feature, columns=list(X.columns))

for seed in range(SEED):
    X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                        test_size=0.3,
                                                        random_state = seed)
    X_train, y_train = pipeline.fit_resample(X_train,y_train)

    rf = RandomForestClassifier(class_weight = 'balanced', 
                                    random_state=seed)
        #search across 100 different combinations, and use all available cores
    rf_search = RandomizedSearchCV(estimator = rf, 
                                       param_distributions = random_grid, 
                                       n_iter = 1000, cv = 5, verbose=0,
                                       scoring='roc_auc',
                                       random_state=seed, n_jobs = -1)
        #training and search for the best estimators
    rf_search.fit(X_train, y_train)

    best_rf = rf_search.best_estimator_
    assess(best_rf, X_train, y_train, X_test, y_test)
    
    test_prediction = best_rf.predict(X_test)
    train_probs = best_rf.predict_proba(X_train)[:,1]
    test_probs = best_rf.predict_proba(X_test)[:,1]

    
    report_rf.at[seed,'accuracy'] = accuracy_score(test_prediction, y_test)
    report_rf.at[seed, 'train_ROC'] = roc_auc_score(y_train, train_probs)
    report_rf.at[seed,'test_ROC'] = roc_auc_score(y_test, test_probs)
    report_rf.at[seed,'precision'] = precision_score(test_prediction, y_test)
    report_rf.at[seed,'recall'] = recall_score(test_prediction, y_test)
    report_rf.at[seed,'F1_score'] = f1_score(test_prediction, y_test)
    importance = list(best_rf.feature_importances_)
    imp_feature_rf.loc[len(imp_feature_rf)]=importance


#training personal centre data

report = np.zeros((SEED,6))
report_names = ['accuracy', 'precision','recall', 'F1_score', 'train_ROC', 'test_ROC']
report_rf2 = pd.DataFrame(report,columns=report_names)
imp_feature = np.zeros((0,X2.shape[1]))
imp_feature_rf2 = pd.DataFrame(imp_feature,  columns=list(X2.columns))

for seed in range(SEED):
    X_train, X_test, y_train, y_test = train_test_split(X2, y2, 
                                                        test_size=0.3,
                                                        random_state = seed)
    X_train, y_train = pipeline.fit_resample(X_train,y_train)
    rf = RandomForestClassifier(class_weight = classweight_cal(y_train), 
                                    random_state=seed)
        #search across 100 different combinations, and use all available cores
    rf_search = RandomizedSearchCV(estimator = rf, 
                                       param_distributions = random_grid, 
                                       n_iter = 1000, cv = 5, verbose=0,
                                       scoring='roc_auc',
                                       random_state=seed, n_jobs = -1)
        #training and search for the best estimators
    rf_search.fit(X_train, y_train)

    best_rf = rf_search.best_estimator_
    
    test_prediction = best_rf.predict(X_test)
    train_probs = best_rf.predict_proba(X_train)[:,1]
    test_probs = best_rf.predict_proba(X_test)[:,1]
    
    report_rf2.at[seed,'accuracy'] = accuracy_score(test_prediction, y_test)
    report_rf2.at[seed, 'train_ROC'] = roc_auc_score(y_train, train_probs)
    report_rf2.at[seed,'test_ROC'] = roc_auc_score(y_test, test_probs)
    report_rf2.at[seed,'precision'] = precision_score(test_prediction, y_test)
    report_rf2.at[seed,'recall'] = recall_score(test_prediction, y_test)
    report_rf2.at[seed,'F1_score'] = f1_score(test_prediction, y_test)
    importances = list(best_rf.feature_importances_)
    imp_feature_rf2.loc[len(imp_feature_rf2)]=importances


#%%
'''XGBoost model'''
#parameter definition
params = {
        'learning_rate': [0.03, 0.01, 0.003, 0.001],
        'min_child_weight': [1,3, 5,7, 10],
        'gamma': [0, 0.5, 1, 1.5, 2, 2.5, 5],
        'subsample': [0.6, 0.8, 1.0, 1.2, 1.4],
        'colsample_bytree': [0.6, 0.8, 1.0, 1.2, 1.4],
        'max_depth': [3, 4, 5, 6, 7, 8, 9 ,10, 12, 14],
        'reg_lambda':np.array([0.4, 0.6, 0.8, 1, 1.2, 1.4])}

# specific parameters. I set early stopping to avoid overfitting and specify the validation dataset 
fit_params = {'early_stopping_rounds':10,
              'eval_set':[(X_test, y_test)]}

# let's run the optimization
report = np.zeros((SEED,6))
report_names = ['accuracy', 'precision','recall', 'F1_score', 'train_ROC', 'test_ROC']
report_xgb = pd.DataFrame(report,columns=report_names)
imp_feature = np.zeros((0,X_smote.shape[1]))
imp_feature_xgb = pd.DataFrame(imp_feature,columns=X_smote.columns)

for seed in range(SEED):
    X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                        test_size=0.3,
                                                        random_state = seed)
    X_train, y_train = pipeline.fit_resample(X_train,y_train)
    xgb_model = XGBClassifier(classweight = classweight_cal(y_train), seed = seed)
    #search for the best parameter
    xgb_search = RandomizedSearchCV(xgb_model, param_distributions=params, 
                                       n_iter=1000,
                                       scoring="roc_auc", 
                                       n_jobs=-1, verbose=0,
                                       random_state=seed, cv=5 )
    # n_iter : number of iteration
    # scoring : loss 
    # n_jobs : parallel computation if -1 means use all the threads available
    # cv : number of folds of the cross-validation 
    
    #training and search for the best estimators
    xgb_search.fit(X_train, y_train)

    best_xgb = xgb_search.best_estimator_
    
    #assess performance
    test_prediction = best_xgb.predict(X_test)
    train_probs = best_xgb.predict_proba(X_train)[:,1]
    test_probs = best_xgb.predict_proba(X_test)[:,1]
    
    report_xgb.at[seed,'accuracy'] = accuracy_score(test_prediction, y_test)
    report_xgb.at[seed, 'train_ROC'] = roc_auc_score(y_train, train_probs)
    report_xgb.at[seed,'test_ROC'] = roc_auc_score(y_test, test_probs)
    report_xgb.at[seed,'precision'] = precision_score(test_prediction, y_test)
    report_xgb.at[seed,'recall'] = recall_score(test_prediction, y_test)
    report_xgb.at[seed,'F1_score'] = f1_score(test_prediction, y_test)
    importance = list(best_rf.feature_importances_)
    imp_feature_xgb.loc[len(imp_feature_xgb)]=importance


# training personal centre data

report = np.zeros((SEED,6))
report_names = ['accuracy', 'precision','recall', 'F1_score', 'train_ROC', 'test_ROC']
report_xgb2 = pd.DataFrame(report,columns=report_names)
imp_feature = np.zeros((0,X_smote2.shape[1]))
imp_feature_xgb2 = pd.DataFrame(imp_feature,columns=X_smote2.columns)

for seed in range(SEED):
    X_train, X_test, y_train, y_test = train_test_split(X2, y2, 
                                                        test_size=0.3,
                                                        random_state = seed)
    X_train, y_train = pipeline.fit_resample(X_train,y_train)

    xgb_model = XGBClassifier(classweight = classweight_cal(y_train), seed = seed)
    #search for the best parameter
    xgb_search = RandomizedSearchCV(xgb_model, param_distributions=params, 
                                       n_iter=1000,
                                       scoring="roc_auc", 
                                       n_jobs=-1, verbose=0,
                                       random_state=seed, cv=5 )

    xgb_search.fit(X_train, y_train)

    best_xgb = xgb_search.best_estimator_
    
    #assess performance
    test_prediction = best_xgb.predict(X_test)
    train_probs = best_xgb.predict_proba(X_train)[:,1]
    test_probs = best_xgb.predict_proba(X_test)[:,1]
    
    report_xgb2.at[seed,'accuracy'] = accuracy_score(test_prediction, y_test)
    report_xgb2.at[seed, 'train_ROC'] = roc_auc_score(y_train, train_probs)
    report_xgb2.at[seed,'test_ROC'] = roc_auc_score(y_test, test_probs)
    report_xgb2.at[seed,'precision'] = precision_score(test_prediction, y_test)
    report_xgb2.at[seed,'recall'] = recall_score(test_prediction, y_test)
    report_xgb2.at[seed,'F1_score'] = f1_score(test_prediction, y_test)
    importance = list(best_rf.feature_importances_)
    imp_feature_xgb2.loc[len(imp_feature_xgb2)]=importance
    
report_assess(report_xgb, report_xgb2)

report_assess(imp_feature_rf, imp_feature_rf2)
# %%
'''SVC'''
# SVC on IVF centre patients

param_svc = {'C': [0.1, 1, 10, 100],
              'gamma': [1, 0.1, 0.01, 0.001],
              'kernel': ['linear', 'rbf', 'poly', 'sigmoid']}

#SVC on IVF centre patients
report = np.zeros((SEED,6))
report_names = ['accuracy', 'precision','recall', 'F1_score', 'train_ROC', 'test_ROC']
report_svc = pd.DataFrame(report,columns=report_names)

for seed in range(SEED):
    X_train, X_test, y_train, y_test = train_test_split(X_smote, y_smote, 
                                                        test_size=0.3,
                                                        random_state = seed)
    
    svc_model = SVC(random_state=seed, 
                    class_weight='balanced',
                    probability=True)
    
    svc_search = GridSearchCV(svc_model, param_svc,
                                       scoring="roc_auc", 
                                       n_jobs=-1, verbose=0, cv=5 )
    
    #training and search for the best estimators
    svc_search.fit(X_train, y_train)

    best_svc = svc_search.best_estimator_
    
    #assess performance
    test_prediction = best_svc.predict(X_test)
    train_probs = best_svc.predict_proba(X_train)[:,1]
    test_probs = best_svc.predict_proba(X_test)[:,1]
    
    report_svc.at[seed,'accuracy'] = accuracy_score(test_prediction, y_test)
    report_svc.at[seed, 'train_ROC'] = roc_auc_score(y_train, train_probs)
    report_svc.at[seed,'test_ROC'] = roc_auc_score(y_test, test_probs)
    report_svc.at[seed,'precision'] = precision_score(test_prediction, y_test)
    report_svc.at[seed,'recall'] = recall_score(test_prediction, y_test)
    report_svc.at[seed,'F1_score'] = f1_score(test_prediction, y_test)

#SVC on personal centre patients
report = np.zeros((SEED,6))
report_names = ['accuracy', 'precision','recall', 'F1_score', 'train_ROC', 'test_ROC']
report_svc2 = pd.DataFrame(report,columns=report_names)

for seed in range(SEED):
    X_train, X_test, y_train, y_test = train_test_split(X_smote2, y_smote2, 
                                                        test_size=0.3,
                                                        random_state = seed)
    
    svc_model = SVC(random_state=seed, 
                    class_weight='balanced',
                    probability=True)
    svc_search = GridSearchCV(svc_model, param_svc,
                                       scoring="roc_auc", 
                                       n_jobs=-1, verbose=0, cv=5 )
    
    #training and search for the best estimators
    svc_search.fit(X_train, y_train)

    best_svc = svc_search.best_estimator_
    
    #assess performance
    test_prediction = best_svc.predict(X_test)
    train_probs = best_svc.predict_proba(X_train)[:,1]
    test_probs = best_svc.predict_proba(X_test)[:,1]
    
    report_svc2.at[seed,'accuracy'] = accuracy_score(test_prediction, y_test)
    report_svc2.at[seed, 'train_ROC'] = roc_auc_score(y_train, train_probs)
    report_svc2.at[seed,'test_ROC'] = roc_auc_score(y_test, test_probs)
    report_svc2.at[seed,'precision'] = precision_score(test_prediction, y_test)
    report_svc2.at[seed,'recall'] = recall_score(test_prediction, y_test)
    report_svc2.at[seed,'F1_score'] = f1_score(test_prediction, y_test)

#%%
'''kNN'''

#define model and hyperparameters

param_kNN = {'leaf_size': list(range(1,50)),
             'n_neighbors': list(range(1,30)),
             'p': [1,2]}

### kNN on IVF centre patients
report = np.zeros((SEED,6))
report_names = ['accuracy', 'precision','recall', 'F1_score', 'train_ROC', 'test_ROC']
report_knn = pd.DataFrame(report,columns=report_names)

for seed in range(SEED):
    X_train, X_test, y_train, y_test = train_test_split(X_rus, y_rus, 
                                                        test_size=0.3,
                                                        random_state = seed)
    
    kNN_model = KNeighborsClassifier()

    kNN_search = GridSearchCV(kNN_model, param_kNN,
                          scoring="roc_auc", 
                          n_jobs=-1, verbose=0, cv=5)
    
    #training and search for the best estimators
    kNN_search.fit(X_train, y_train)

    best_kNN = kNN_search.best_estimator_
    
    #assess performance
    test_prediction = best_kNN.predict(X_test)
    train_probs = best_kNN.predict_proba(X_train)[:,1]
    test_probs = best_kNN.predict_proba(X_test)[:,1]
    
    report_knn.at[seed,'accuracy'] = accuracy_score(test_prediction, y_test)
    report_knn.at[seed, 'train_ROC'] = roc_auc_score(y_train, train_probs)
    report_knn.at[seed,'test_ROC'] = roc_auc_score(y_test, test_probs)
    report_knn.at[seed,'precision'] = precision_score(test_prediction, y_test)
    report_knn.at[seed,'recall'] = recall_score(test_prediction, y_test)
    report_knn.at[seed,'F1_score'] = f1_score(test_prediction, y_test)


###kNN on personal centre patients
report = np.zeros((SEED,6))
report_names = ['accuracy', 'precision','recall', 'F1_score', 'train_ROC', 'test_ROC']
report_knn2 = pd.DataFrame(report,columns=report_names)

for seed in range(SEED):
    X_train, X_test, y_train, y_test = train_test_split(X_rus2, y_rus2,
                                                        test_size=0.3,
                                                        random_state = seed)
    
    kNN_model = KNeighborsClassifier()

    kNN_search = GridSearchCV(kNN_model, param_kNN,
                          scoring="roc_auc", 
                          n_jobs=-1, verbose=0, cv=5)
    
    #training and search for the best estimators
    kNN_search.fit(X_train, y_train)

    best_kNN = kNN_search.best_estimator_
    
    #assess performance
    test_prediction = best_kNN.predict(X_test)
    train_probs = best_kNN.predict_proba(X_train)[:,1]
    test_probs = best_kNN.predict_proba(X_test)[:,1]
    
    report_knn2.at[seed,'accuracy'] = accuracy_score(test_prediction, y_test)
    report_knn2.at[seed, 'train_ROC'] = roc_auc_score(y_train, train_probs)
    report_knn2.at[seed,'test_ROC'] = roc_auc_score(y_test, test_probs)
    report_knn2.at[seed,'precision'] = precision_score(test_prediction, y_test)
    report_knn2.at[seed,'recall'] = recall_score(test_prediction, y_test)
    report_knn2.at[seed,'F1_score'] = f1_score(test_prediction, y_test)
    
#%%
'''Logistic Regression'''

# define grid search
grid = {'solver': ['newton-cg', 'lbfgs', 'liblinear'],
                'penalty': ['l1'],
                'C': [100, 10, 1.0, 0.1, 0.01]}

# Logistic regression on IVF patients
report = np.zeros((SEED,6))
report_names = ['accuracy', 'precision','recall', 'F1_score', 'train_ROC', 'test_ROC']
report_lg = pd.DataFrame(report,columns=report_names)

for seed in range(SEED):
    X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                        test_size=0.3,
                                                        random_state = seed)
    X_train, y_train = pipeline.fit_resample(X_train, y_train)
    #X_train, y_train = smote.fit_resample(X_train, y_train)
    lg = LogisticRegression(random_state = 5, 
                    class_weight = 'balanced')

    lg_search = GridSearchCV(estimator=lg, param_grid=grid, n_jobs=-1, 
                             cv=5, scoring='roc_auc',error_score=0)
    
    lg_search.fit(X_train, y_train)

    best_lg = lg_search.best_estimator_
    test_prediction = best_lg.predict(X_test)
    train_probs = best_lg.predict_proba(X_train)[:,1]
    test_probs = best_lg.predict_proba(X_test)[:,1]
    
    report_lg.at[seed,'accuracy'] = accuracy_score(test_prediction, y_test)
    report_lg.at[seed, 'train_ROC'] = roc_auc_score(y_train, train_probs)
    report_lg.at[seed,'test_ROC'] = roc_auc_score(y_test, test_probs)
    report_lg.at[seed,'precision'] = precision_score(test_prediction, y_test)
    report_lg.at[seed,'recall'] = recall_score(test_prediction, y_test)
    report_lg.at[seed,'F1_score'] = f1_score(test_prediction, y_test)



#Logistic regression on personal centre patients
   
report = np.zeros((SEED,6))
report_names = ['accuracy', 'precision','recall', 'F1_score', 'train_ROC', 'test_ROC']
report_lg2 = pd.DataFrame(report,columns=report_names)

for seed in range(SEED):
    X_train, X_test, y_train, y_test = train_test_split(X2, y2, 
                                                        test_size=0.3,
                                                        random_state = seed)
    X_train, y_train = pipeline.fit_resample(X_train, y_train)
    lg = LogisticRegression(random_state = 5, 
                    class_weight = 'balanced')

    lg_search = GridSearchCV(estimator=lg, param_grid=grid, n_jobs=-1, 
                             cv=5, scoring='roc_auc',error_score=0)
    
    lg_search.fit(X_train, y_train)

    best_lg = lg_search.best_estimator_
    test_prediction_lg2 = best_lg.predict(X_test)
    train_probs_lg2 = best_lg.predict_proba(X_train)[:,1]
    test_probs_lg2 = best_lg.predict_proba(X_test)[:,1]
    
    report_lg2.loc[seed,'accuracy'] = accuracy_score(test_prediction_lg2, y_test)
    report_lg2.loc[seed, 'train_ROC'] = roc_auc_score(y_train, train_probs_lg2)
    report_lg2.loc[seed,'test_ROC'] = roc_auc_score(y_test, test_probs_lg2)
    report_lg2.loc[seed,'precision'] = precision_score(test_prediction_lg2, y_test)
    report_lg2.at[seed,'recall'] = recall_score(test_prediction_lg2, y_test)
    report_lg2.at[seed,'F1_score'] = f1_score(test_prediction_lg2, y_test)

    
report_assess(report_lg, report_lg2)

#%%
'''Gaussian Naive Bayes'''


param_gnb = {'var_smoothing': [1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7,
                                1e-8, 1e-9, 1e-10, 1e-11, 1e-12, 1e-13, 1e-14,
                                1e-15]}
report = np.zeros((SEED,6))
report_names = ['accuracy', 'precision','recall', 'F1_score', 'train_ROC', 'test_ROC']
report_gnb = pd.DataFrame(report,columns=report_names)

for seed in range(SEED):
    X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                        test_size=0.3,
                                                        random_state = seed)
    X_train, y_train = pipeline.fit_resample(X_train, y_train)
    gnb  = GaussianNB()
    
    gnb_search = GridSearchCV(gnb, param_grid=param_gnb, n_jobs=-1,
                             cv=5, scoring='roc_auc',error_score=0)

    gnb_search.fit(X_train, y_train)

    best_gnb = gnb_search.best_estimator_
    test_prediction = best_gnb.predict(X_test)
    train_probs = best_gnb.predict_proba(X_train)[:,1]
    test_probs = best_gnb.predict_proba(X_test)[:,1]
    
    report_gnb.at[seed,'accuracy'] = accuracy_score(test_prediction, y_test)
    report_gnb.at[seed, 'train_ROC'] = roc_auc_score(y_train, train_probs)
    report_gnb.at[seed,'test_ROC'] = roc_auc_score(y_test, test_probs)
    report_gnb.at[seed,'precision'] = precision_score(test_prediction, y_test)
    report_gnb.at[seed,'recall'] = recall_score(test_prediction, y_test)
    report_gnb.at[seed,'F1_score'] = f1_score(test_prediction, y_test)
    


#training on personal centre patients
report = np.zeros((SEED,6))
report_names = ['accuracy', 'precision','recall', 'F1_score', 'train_ROC', 'test_ROC']
report_gnb2 = pd.DataFrame(report,columns=report_names)

for seed in range(SEED):
    X_train, X_test, y_train, y_test = train_test_split(X2, y2, 
                                                        test_size=0.3,
                                                        random_state = seed)
    X_train, y_train = pipeline.fit_resample(X_train, y_train)
    gnb  = GaussianNB()
    
    gnb_search = GridSearchCV(gnb, param_grid=param_gnb, n_jobs=-1,
                             cv=5, scoring='roc_auc',error_score=0)

    gnb_search.fit(X_train, y_train)

    best_gnb = gnb_search.best_estimator_
    test_prediction = best_gnb.predict(X_test)
    train_probs = best_gnb.predict_proba(X_train)[:,1]
    test_probs = best_gnb.predict_proba(X_test)[:,1]
    
    report_gnb2.at[seed,'accuracy'] = accuracy_score(test_prediction, y_test)
    report_gnb2.at[seed, 'train_ROC'] = roc_auc_score(y_train, train_probs)
    report_gnb2.at[seed,'test_ROC'] = roc_auc_score(y_test, test_probs)
    report_gnb2.at[seed,'precision'] = precision_score(test_prediction, y_test)
    report_gnb2.at[seed,'recall'] = recall_score(test_prediction, y_test)
    report_gnb2.at[seed,'F1_score'] = f1_score(test_prediction, y_test)
    
report_assess(report_gnb, report_gnb2)




