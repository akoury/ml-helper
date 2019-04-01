#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  11 09:19:35 2019

@author: alejandrokoury
"""

import timeit
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.base import clone
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from scipy.stats import chi2_contingency
from scipy.stats.mstats import winsorize
from sklearn.pipeline import make_pipeline
from sklearn.compose import ColumnTransformer
from sklearn.neighbors import LocalOutlierFactor
from sklearn.model_selection import cross_val_score, RandomizedSearchCV, TimeSeriesSplit, StratifiedKFold

class Helper:
    
    def __init__(self, keys):
        self.keys = keys
        self.SEED = keys['SEED']
        self.TARGET = keys['TARGET']
        self.METRIC = keys['METRIC']
        self.TIMESERIES = keys['TIMESERIES']
        self.SPLITS = keys['SPLITS']

    def get_keys(self):
        return self.keys
    
    # Data Exploration
    
    def missing_data(self, df):
        total = df.isnull().sum().sort_values(ascending=False)
        percent = (df.isnull().sum()/df.isnull().count()).sort_values(ascending=False)
        return pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
    
    
    def types(self, df, types, exclude = None):
        types = df.select_dtypes(include=types)
        excluded = [self.TARGET]
        if exclude:
            for i in exclude:
                excluded.append(i)
        cols = [col for col in types.columns if col not in excluded]
        return df[cols]

    def numericals(self, df, exclude = None):
        return self.types(df, [np.number], exclude)

    def categoricals(self, df, exclude = None):
        return self.types(df, ['category', object], exclude)
    
    def correlated(self, df, threshold = 0.9):
        categoric = self.categorical_correlated(df, threshold)
        numeric = self.numerical_correlated(df, threshold)

        plt.figure(figsize=(12,10))
        sns.heatmap(categoric[1],cbar=True,fmt =' .2f', annot=True, cmap='viridis').set_title('Categorical Correlation', fontsize=30)

        plt.figure(figsize=(12,10))
        sns.heatmap(numeric[1],cbar=True,fmt =' .2f', annot=True, cmap='viridis').set_title('Numerical Correlation', fontsize=30)

        correlated_cols = categoric[0] + numeric[0]

        if(len(correlated_cols) > 0):
            print('The following columns are correlated with a threshold of ' + str(threshold) + ': ' + str(correlated_cols))

            if self.TARGET in correlated_cols:
                print('The target variable is correlated, consider removing its correlated counterpart')
                correlated_cols.remove(self.TARGET)
        else:
            print('No correlated columns for the  ' + str(threshold) + ' threshold')

        return correlated_cols
    
    def numerical_correlated(self, df, threshold=0.9):
        corr_matrix = np.absolute(df.select_dtypes(include=[np.number]).corr(method='spearman')).abs()
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))
        return [column for column in upper.columns if any(abs(upper[column]) > threshold)], corr_matrix

    def cramers_v(self, x, y):
        confusion_matrix = pd.crosstab(x, y)
        chi2 = chi2_contingency(confusion_matrix)[0]
        n = confusion_matrix.sum().sum()
        phi2 = chi2/n
        r, k = confusion_matrix.shape
        phi2corr = max(0, phi2-((k-1)*(r-1))/(n-1))
        rcorr = r-((r-1)**2)/(n-1)
        kcorr = k-((k-1)**2)/(n-1)
        return np.sqrt(phi2corr/min((kcorr-1), (rcorr-1)))

    def categorical_correlated(self, df, threshold=0.9):
        columns = df.select_dtypes(include=['object', 'category']).columns.tolist()
        corr = pd.DataFrame(index=columns, columns=columns)
        for i in range(0, len(columns)):
            for j in range(i, len(columns)):
                if i == j:
                    corr[columns[i]][columns[j]] = 1.0
                else:
                    cell = self.cramers_v(df[columns[i]], df[columns[j]])
                    corr[columns[i]][columns[j]] = cell
                    corr[columns[j]][columns[i]] = cell
        corr.fillna(value=np.nan, inplace=True)
        upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(np.bool))
        return [column for column in upper.columns if any(abs(upper[column]) > threshold)], corr
    
    def under_represented(self, df, threshold = 0.99):
        under_rep = []
        for column in df:
            counts = df[column].value_counts()
            majority_freq = counts.iloc[0]
            if (majority_freq / len(df)) > threshold:
                under_rep.append(column)

        if not under_rep:
            print('No underrepresented features')
        else:
            if self.TARGET in under_rep:
                print('The target variable is underrepresented, consider rebalancing')
                under_represented.remove(self.TARGET)
            print(str(under_rep) + ' underrepresented')

        return under_rep
    
    def feature_importance(self, df, model):
        X, y = self.split_x_y(df)
        model.fit(X, y)
        importances = model.feature_importances_
        std = np.std([tree.feature_importances_ for tree in model.estimators_],axis=0)
        indices = np.argsort(importances)

        print("Feature ranking:")
        plt.figure(figsize=(16, 14))
        plt.title("Feature importances")
        plt.barh(range(X.shape[1]), importances[indices],color="r", xerr=std[indices], align="center")
        plt.yticks(range(X.shape[1]), [list(X)[i] for i in indices])
        plt.ylim([-1, X.shape[1]])
        plt.show()

    def plot_pca_components(self, df, variance = 0.9):
        X, y = self.split_x_y(df)
        pca = PCA().fit(X)

        plt.figure()
        plt.plot(np.cumsum(pca.explained_variance_ratio_))
        plt.xlabel('Number of Components')
        plt.ylabel('Variance (%)')
        plt.show()
    
    # Data Preparation
    
    def convert_to_category(self, df, cols):
        for i in cols:
            df[i] = df[i].astype('category')
        return df
    
    def drop_columns(self, df, cols):
        return df.drop(df[cols], axis=1)
    
    # Modeling
    
    def cv_evaluate(self, df, model, splits = None, pipe = None, grid = None):
        if splits is None:
            splits = self.SPLITS
        
        X, y = self.split_x_y(df)
        
        if self.TIMESERIES:
            folds = TimeSeriesSplit(n_splits = splits)
        else:
            folds = StratifiedKFold(n_splits = splits, shuffle = True, random_state=self.SEED)

        if pipe:
            pipe_cv = clone(pipe)
            pipe_cv.steps.append((model['name'], model['model']))
            model = pipe_cv

        if grid:
            model = RandomizedSearchCV(model, grid, scoring = self.METRIC, cv = folds, n_iter = 10, refit=True, return_train_score = False, error_score=0.0, random_state = self.SEED)
            model.fit(X, y)
            scores = model.cv_results_['mean_test_score']
        else:
            scores = cross_val_score(model, X, y, scoring = self.METRIC, cv = folds)
            
        return scores, model
    
    def pipeline(self, df, models, pipe, all_scores = pd.DataFrame(), splits = None, note = ''):
        if splits is None:
            splits = self.SPLITS
        
        for model in models:
            try:
                start = timeit.default_timer()

                scores, cv_model = self.cv_evaluate(df.copy(), model, pipe = pipe, splits = splits)

            except Exception as error:
                note = 'Error: ' + str(error)
                print(note)
                scores = np.array([0])

            all_scores = self.score(model['name'], scores, timeit.default_timer(), start, cv_model, note, all_scores)
            
        self.show_scores(all_scores)
            
        return all_scores
    
    def predict(self, df, holdout, pipe):
        X_train, y_train = self.split_x_y(df)
        pipe.fit(X_train, y_train)

        X, y = self.split_x_y(holdout)
        
        return y, pipe.predict(X)
    
    # Others
    
    def split_x_y(self, df):
        return df.loc[:, df.columns != self.TARGET], df.loc[:, self.TARGET]
    
    def flatten(self, pipe):
        flat = []
        for i in pipe:
            if isinstance(i,list): flat.extend(self.flatten(i))
            else: flat.append(i)
        return flat

    def pipe_steps(self, pipe):
        return self.flatten([x[0] if not isinstance(x[1], ColumnTransformer) else [list(i[1].named_steps.keys()) for ind,i in enumerate(x[1].transformers)] for x in pipe.steps])
    
    def score(self, model, scores, stop, start, pipe, note = '', all_scores = pd.DataFrame()):
        if len(all_scores) == 0:
            all_scores  = pd.DataFrame(columns = ['Model', 'Mean', 'CV Score', 'Time', 'Cumulative', 'Pipe', 'Steps', 'Note'])

        if len(scores[scores > 0]) == 0:
            mean = 0
            std = 0
        else:
            mean = np.mean(scores[scores > 0])
            std = np.std(scores[scores > 0])
        
        cumulative = stop - start
        if len(all_scores[all_scores['Model'] == model]) > 0:
            cumulative += all_scores[all_scores['Model'] == model].tail(1)['Time'].values[0]
            
        all_scores = all_scores.append({'Model': model, 'Mean': mean, 'CV Score': '{:.3f} +/- {:.3f}'.format(mean, std), 'Time': stop - start, 'Cumulative': cumulative, 'Pipe': pipe, 'Steps': self.pipe_steps(pipe), 'Note': note}, ignore_index=True)
    
        return all_scores.loc[all_scores.astype(str).drop_duplicates(subset=['Model', 'CV Score', 'Steps'], keep='first').index]

    def show_scores(self, all_scores):
        pd.set_option('max_colwidth', -1)
        display(all_scores.loc[:, ~all_scores.columns.isin(['Mean', 'Pipe', 'Cumulative'])])
        
    def plot_models(self, all_scores):
        plt.figure(figsize=(16, 8))
        ax = sns.lineplot(x="Cumulative", y="Mean", hue="Model", style="Model", markers=True, dashes=False, data=all_scores)
        label = str(self.METRIC) + ' Score'
        ax.set(ylabel=label, xlabel='Time', ylim=(0, 1))
        
    def best_pipeline(self, all_scores):
        return all_scores.sort_values(by=['Mean'], ascending = False).iloc[0]['Pipe']