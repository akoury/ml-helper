#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  11 09:19:35 2019

@author: alejandrokoury
"""

SEED = 1
TARGET_VARIABLE = 'cnt'
SPLITS = 4
ESTIMATORS = 50
METRIC = 'r2'
TIMESERIES = True

if METRIC == 'r2':
    from sklearn.metrics import r2_score as metric_scorer
else:
    from sklearn.metrics import accuracy_score as metric_scorer


import numpy as np
import pandas as pd
import seaborn as sns
from tempfile import mkdtemp
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from scipy.stats import chi2_contingency
from scipy.stats.mstats import winsorize
from sklearn.pipeline import make_pipeline
from sklearn.neighbors import LocalOutlierFactor
from sklearn.model_selection import cross_val_score, RandomizedSearchCV, TimeSeriesSplit, StratifiedKFold


def missing_data(df):
    total = df.isnull().sum().sort_values(ascending=False)
    percent = (df.isnull().sum()/df.isnull().count()).sort_values(ascending=False)
    return pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])

def convert_to_category(df, cols):
    for i in cols:
        df[i] = df[i].astype('category')
    return df

def drop_columns(df, cols):
    return df.drop(df[cols], axis=1)

def types(df, types, exclude = None):
    types = df.select_dtypes(include=types)
    excluded = [TARGET_VARIABLE]
    if exclude:
        for i in exclude:
            excluded.append(i)
    cols = [col for col in types.columns if col not in excluded]
    return df[cols]

def numericals(df, exclude = None):
    return types(df, [np.number], exclude)

def categoricals(df, exclude = None):
    return types(df, ['category', object], exclude)

def numerical_correlated(df, threshold=0.9):
    corr_matrix = np.absolute(df.select_dtypes(include=[np.number]).corr(method='spearman')).abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))
    return [column for column in upper.columns if any(abs(upper[column]) > threshold)], corr_matrix

def cramers_v(x, y):
    confusion_matrix = pd.crosstab(x, y)
    chi2 = chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum().sum()
    phi2 = chi2/n
    r, k = confusion_matrix.shape
    phi2corr = max(0, phi2-((k-1)*(r-1))/(n-1))
    rcorr = r-((r-1)**2)/(n-1)
    kcorr = k-((k-1)**2)/(n-1)
    return np.sqrt(phi2corr/min((kcorr-1), (rcorr-1)))

def categorical_correlated(df, threshold=0.9):
    columns = df.select_dtypes(include=['object', 'category']).columns.tolist()
    corr = pd.DataFrame(index=columns, columns=columns)
    for i in range(0, len(columns)):
        for j in range(i, len(columns)):
            if i == j:
                corr[columns[i]][columns[j]] = 1.0
            else:
                cell = cramers_v(df[columns[i]], df[columns[j]])
                corr[columns[i]][columns[j]] = cell
                corr[columns[j]][columns[i]] = cell
    corr.fillna(value=np.nan, inplace=True)
    upper = corr.where(
    np.triu(np.ones(corr.shape), k=1).astype(np.bool))
    return [column for column in upper.columns if any(abs(upper[column]) > threshold)], corr

def correlated(df, threshold = 0.9):
    categoric = categorical_correlated(df, threshold)
    numeric = numerical_correlated(df, threshold)
    
    plt.figure(figsize=(12,10))
    sns.heatmap(categoric[1],cbar=True,fmt =' .2f', annot=True, cmap='viridis').set_title('Categorical Correlation', fontsize=30)
    
    plt.figure(figsize=(12,10))
    sns.heatmap(numeric[1],cbar=True,fmt =' .2f', annot=True, cmap='viridis').set_title('Numerical Correlation', fontsize=30)
    
    correlated_cols = categoric[0] + numeric[0]
    
    if(len(correlated_cols) > 0):
        print('The following columns are correlated with a threshold of ' + str(threshold) + ': ' + str(correlated_cols))
    else:
        print('No correlated columns for the  ' + str(threshold) + ' threshold')
    
    return correlated_cols, categoric[1], numeric[1]

def winsorize_data(df, train_df, cols):
    for col in cols:
        train_df[col] = winsorize(train_df[col], limits = [0.01, 0.01])
        df[df[col] > max(train_df[col])][col] = max(train_df[col])
        df[df[col] < min(train_df[col])][col] = min(train_df[col])
    return df

def lof(df, training_df):
    lof = LocalOutlierFactor(n_neighbors=20, contamination='auto')
    y_pred = lof.fit_predict(training_df)
    outliers = np.where(y_pred == -1)
    print('Removing ' + str(len(outliers[0])) + ' records')
    return df.drop(outliers[0])

def one_hot_encode(df, cols):
    for i in cols:
        dummies = pd.get_dummies(df[i], prefix=i, drop_first = False)
        df = pd.concat([df, dummies], axis = 1)
        df = df.drop(i, axis = 1)
    
    return df

def under_represented_features(df, threshold = 0.99, holdout_df = None):
    under_rep = []
    for column in df:
        counts = df[column].value_counts()
        majority_freq = counts.iloc[0]
        if (majority_freq / len(df)) > threshold:
            under_rep.append(column)
            
    if not under_rep:
        print('No underrepresented features')
    else:
        if TARGET_VARIABLE in under_rep:
            print('The target variable is underrepresented, consider rebalancing')
            under_represented.remove(TARGET_VARIABLE)
        print(str(under_rep) + ' underrepresented, removing')
    
    df = drop_columns(df, under_rep)
    
    if holdout_df is not None:
        return df, drop_columns(holdout_df, under_rep)
    
    return df

def feature_importance(df, model):
    acc, scores, model = cv_evaluate(df, model = model)
    importances = model.feature_importances_
    std = np.std([tree.feature_importances_ for tree in model.estimators_],axis=0)
    indices = np.argsort(importances)
    
    X = df.loc[:, df.columns != TARGET_VARIABLE]
    print("Feature ranking:")
    plt.figure(figsize=(16, 14))
    plt.title("Feature importances")
    plt.barh(range(X.shape[1]), importances[indices],color="r", xerr=std[indices], align="center")
    plt.yticks(range(X.shape[1]), [list(df.loc[:, df.columns != TARGET_VARIABLE])[i] for i in indices])
    plt.ylim([-1, X.shape[1]])
    plt.show()
    
def plot_pca_components(df, variance = 0.9):
    X = df.loc[:, df.columns != TARGET_VARIABLE]
    y = df.loc[:, TARGET_VARIABLE]
    pca = PCA().fit(X)

    plt.figure()
    plt.plot(np.cumsum(pca.explained_variance_ratio_))
    plt.xlabel('Number of Components')
    plt.ylabel('Variance (%)')
    plt.show()

def cv_evaluate(df, model, splits = SPLITS, transformers = None, grid = None):
    X = df.loc[:, df.columns != TARGET_VARIABLE]
    y = df.loc[:, TARGET_VARIABLE]
    if TIMESERIES:
        folds = TimeSeriesSplit(n_splits = splits)
    else:
        folds = StratifiedKFold(n_splits = splits, shuffle = True, random_state=SEED)
    
    train_size = int(len(df) * 0.85)
    X_train, X_validate, y_train, y_validate = X[0:train_size], X[train_size:len(df)], y[0:train_size], y[train_size:len(df)]
    
    if transformers:
        cachedir = mkdtemp()
        model = make_pipeline(model, memory = cachedir)
        for ind,i in enumerate(transformers):
            model.steps.insert(ind,[str(ind+1),i])

    if grid:
        model = RandomizedSearchCV(model, grid, scoring = METRIC, cv = folds, n_iter = 20, refit=True, return_train_score = False, error_score=0.0, random_state = SEED)
        model.fit(X_train, y_train)
        scores = model.cv_results_['mean_test_score']
    else:
        scores = cross_val_score(model, X_train, y_train, scoring = METRIC, cv = folds)
        model.fit(X_train, y_train)

    pred = model.predict(X_validate)
    final_score = metric_scorer(y_validate, pred)
    
    return final_score, scores, model

def feature_engineering_pipeline(df, models, transformers, splits = SPLITS):
    all_scores  = pd.DataFrame(columns = ['Model', 'Function', 'CV Score', 'Holdout Score', 'Difference', 'Outcome'])

    for model in models:
        top_cv_score, cv_scores, cv_model = cv_evaluate(df, model = model['model'], splits = splits)
        model['score'] = best_score = top_cv_score
        model['transformers'] = []
        all_scores = all_scores.append({'Model': model['name'], 'Function':'base_score','CV Score': '{:.2f} +/- {:.02}'.format(np.mean(cv_scores[cv_scores > 0.0]),np.std(cv_scores[cv_scores > 0.0])),'Holdout Score': top_cv_score, 'Difference': 0, 'Outcome': 'Base ' + model['name']}, ignore_index=True)
        
        for transformer in transformers:
            engineered_data = df.copy()
            outcome = 'Rejected'
            
            try:
                top_transformer_score, transformer_scores, cv_model = cv_evaluate(engineered_data, model = model['model'], transformers = [transformer['transformer']], splits = splits)
                difference = (top_transformer_score - best_score)
                
                if difference > 0:
                    model['transformers'] = [i for i in model['transformers'] if i['name'] != transformer['name']]
                    model['transformers'].append(transformer['transformer'])
                    outcome = 'Accepted'
                
                mean = np.mean(transformer_scores[transformer_scores > 0.0])
                std = np.std(transformer_scores[transformer_scores > 0.0])
                if np.isnan(mean) or np.isnan(std):
                    mean = 0.00
                    std = 0.00

                score = {'Model': model['name'], 'Function':transformer['name'],'CV Score': '{:.2f} +/- {:.02}'.format(mean,std),'Holdout Score': top_transformer_score, 'Difference': difference, 'Outcome': outcome}

            except: 
                score = {'Model': model['name'], 'Function':transformer['name'],'CV Score': '0.00 +/- 0.00','Holdout Score': 0, 'Difference': 0, 'Outcome': 'Error'}
        
            all_scores = all_scores.append(score, ignore_index=True)
    return create_pipelines(models), all_scores

def create_pipelines(pipes):
    cachedir = mkdtemp()
    for item in pipes:
        item['pipeline'] = make_pipeline(*item['transformers'], item['model'], memory = cachedir)
    
    return sorted(pipes, key=lambda k: k['score'], reverse = True)