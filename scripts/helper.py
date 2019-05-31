from __future__ import division
import pandas as pd
import numpy as np
import os
import os.path
import matplotlib.pyplot as plt
import pylab as pl
from datetime import timedelta
import random
from scipy import optimize
import time
import seaborn as sns
from sklearn.metrics import roc_curve, auc, classification_report, confusion_matrix
from sklearn import preprocessing, svm, metrics, tree, decomposition, svm
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier, BaggingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import ParameterGrid, train_test_split
from sklearn.metrics import *
from sklearn.preprocessing import StandardScaler
import itertools


# modeling helper functions

def define_clfs_params(grid_size):

    '''
    This functions defines parameter grid for all the classifiers
    Inputs:
        grid_size: how big of a grid do you want. it can be test, small, or large
    Returns:
        a set of model and parameters
    Raises:
        KeyError: Raises an exception.
    '''

    clfs = {'RF': RandomForestClassifier(n_estimators=50, n_jobs=-1),
            'B': BaggingClassifier(),
            'LR': LogisticRegression(penalty='l1', C=1e5),
            'SVM': svm.LinearSVC(),
            'GB': GradientBoostingClassifier(learning_rate=0.05, subsample=0.5, max_depth=6, n_estimators=10),
            'DT': DecisionTreeClassifier(),
            'KNN': KNeighborsClassifier(n_neighbors=3) 
            }

    large_grid = { 
    'RF': {'n_estimators': [1,10,100,1000,10000], 
           'max_depth': [1,5,10,20,50,100], 
           'max_features': ['sqrt','log2'],
           'min_samples_split': [2,5,10], 
           'n_jobs': [-1]},
    'B': {'n_estimators': [1,10,100,1000,10000]},
    'LR': {'penalty': ['l1','l2'], 
           'C': [0.00001,0.0001,0.001,0.01,0.1,1,10]},
    'SVM': {'C': [0.00001,0.0001,0.001,0.01,0.1,1,10]},
    'GB': {'n_estimators': [1,10,100,1000,10000], 
           'learning_rate' : [0.001,0.01,0.05,0.1,0.5],
           'subsample' : [0.1,0.5,1.0], 
           'max_depth': [1,3,5,10,20,50,100]},
    'DT': {'criterion': ['gini', 'entropy'], 
           'max_depth': [1,5,10,20,50,100], 
           'max_features': [None, 'sqrt','log2'],
           'min_samples_split': [2,5,10]},
    'KNN' :{'n_neighbors': [1,5,10,25,50,100],
            'weights': ['uniform','distance'],
            'algorithm': ['auto','ball_tree','kd_tree']}
           }

    small_grid = { 
    'RF':{'n_estimators': [100, 10000], 
          'max_depth': [5,50], 
          'max_features': ['sqrt','log2'],
          'min_samples_split': [2,10], 
          'n_jobs':[-1]},
    'B': {'n_estimators': [1,10,100,1000,10000]},
    'LR': {'penalty': ['l1','l2'], 
           'C': [0.00001,0.001,0.1,1,10]},
    'SVM' :{'C': [0.00001,0.0001,0.001,0.01,0.1,1,10]},
    'GB': {'n_estimators': [100, 10000], 
           'learning_rate' : [0.001,0.1,0.5],
           'subsample' : [0.1,0.5,1.0], 
           'max_depth': [5,50]},
    'DT': {'criterion': ['gini', 'entropy'], 
           'max_depth': [1,5,10,20,50,100], 
           'max_features': [None,'sqrt','log2'],
           'min_samples_split': [2,5,10]},
    'KNN': {'n_neighbors': [5,10],
            'weights': ['uniform'],
            'algorithm': ['auto']}
           }
    
    test_grid = { 
    'RF': {'n_estimators': [1], 
           'max_depth': [1], 
           'max_features': ['sqrt'],
           'min_samples_split': [10], 
           'n_jobs': [-1]},
    'B': {'n_estimators': [1]},
    'LR': {'penalty': ['l1'], 
           'C': [0.01]},
    'SVM': {'C': [0.01]},
    'GB': {'n_estimators': [1], 
           'learning_rate' : [0.1],
           'subsample' : [0.5], 
           'max_depth': [1]},
    'DT': {'criterion': ['gini'], 
           'max_depth': [1], 
           'max_features': [None],
           'min_samples_split': [10]},
    'KNN': {'n_neighbors': [5],
            'weights': ['uniform'],
            'algorithm': ['auto']}
           }

    med_grid = { 
    'RF':{'n_estimators': [1, 10, 100, 1000], 
          'max_depth': [5,50], 
          'max_features': ['sqrt','log2'],
          'min_samples_split': [2,10], 
          'n_jobs':[-1]},
    'B': {'n_estimators': [1,10,100,1000]},
    'LR': {'penalty': ['l1','l2'], 
           'C': [0.00001,0.001,0.1,1,10]},
    'SVM' :{'C': [0.00001,0.0001,0.001,0.01,0.1,1,10]},
    'GB': {'n_estimators': [1, 50], 
           'learning_rate' : [0.1, 0.5],
           'subsample' : [0.1, 1.0], 
           'max_depth': [1, 5, 10]},
    'DT': {'criterion': ['gini', 'entropy'], 
           'max_depth': [1,5,10,20,50,100], 
           'max_features': [None,'sqrt','log2'],
           'min_samples_split': [2,5,10]},
    'KNN': {'n_neighbors': [5],
            'weights': ['uniform'],
            'algorithm': ['auto']}
           }

    
    if (grid_size == 'large'):
        return clfs, large_grid
    elif (grid_size == 'small'):
        return clfs, small_grid
    elif (grid_size == 'test'):
        return clfs, test_grid
    elif (grid_size == 'med'):
        return clfs, med_grid
    else:
        return 0, 0


# Evaluation functions
# calculate precision, recall and auc metrics


def precision_at_k(y_true, y_scores, k):
    '''
    '''
    y_scores_sorted, y_true_sorted = joint_sort_descending(np.array(y_scores), np.array(y_true))
    preds_at_k = generate_binary_at_k(y_scores_sorted, k)
    precision = precision_score(y_true_sorted, preds_at_k)
    return precision


def recall_at_k(y_true, y_scores, k):
    '''
    '''
    y_scores_sorted, y_true_sorted = joint_sort_descending(np.array(y_scores), np.array(y_true))
    preds_at_k = generate_binary_at_k(y_scores_sorted, k)
    recall = recall_score(y_true_sorted, preds_at_k)
    return recall


def f1_at_k(y_true, y_scores, k):
    '''
    '''
    y_scores_sorted, y_true_sorted = joint_sort_descending(np.array(y_scores), np.array(y_true))
    preds_at_k = generate_binary_at_k(y_scores_sorted, k)
    f1 = f1_score(y_true_sorted, preds_at_k)
    return f1


def plot_precision_recall_n(y_true, y_prob, model_name):
    '''
    '''
    from sklearn.metrics import precision_recall_curve
    y_score = y_prob
    precision_curve, recall_curve, pr_thresholds = precision_recall_curve(y_true, y_score)
    precision_curve = precision_curve[:-1]
    recall_curve = recall_curve[:-1]
    pct_above_per_thresh = []
    number_scored = len(y_score)
    for value in pr_thresholds:
        num_above_thresh = len(y_score[y_score >= value])
        pct_above_thresh = num_above_thresh / float(number_scored)
        pct_above_per_thresh.append(pct_above_thresh)
    pct_above_per_thresh = np.array(pct_above_per_thresh)
    
    plt.clf()
    fig, ax1 = plt.subplots()
    ax1.plot(pct_above_per_thresh, precision_curve, 'b')
    ax1.set_xlabel('percent of population')
    ax1.set_ylabel('precision', color='b')
    ax2 = ax1.twinx()
    ax2.plot(pct_above_per_thresh, recall_curve, 'r')
    ax2.set_ylabel('recall', color='r')
    ax1.set_ylim([0,1])
    ax1.set_ylim([0,1])
    ax2.set_xlim([0,1])
    
    name = model_name
    plt.title(name)
    plt.show()


def joint_sort_descending(l1, l2):
    '''
    Inputs:
        - l1: numpy array
        - l2: numpy array
    '''
    idx = np.argsort(l1)[::-1]
    return l1[idx], l2[idx]


def generate_binary_at_k(y_scores, k):
    '''
    '''
    cutoff_index = int(len(y_scores) * (k / 100.0))
    predictions_binary = [1 if x < cutoff_index else 0 for x in range(len(y_scores))]
    return predictions_binary


def import_csv(csv_file):
    '''
    This function reads in a csv file and returns a dataframe
    '''
    if not os.path.exists(csv_file):
        print("File path does not exist")
        return None
    df = pd.read_csv(csv_file)
    return df


def explore_data(df):
    '''
    this function will take in a dataframe and print
    the first 10 lines of that dataframe, its shape,
    data types of each column, and the data distribution
    '''
    print("Let's take a look at the first 10 lines of the dataframe!")
    print()
    print(df.head(10))
    print('\n' * 2)
    print("Dataframe's shape: {}".format(df.shape))
    print('\n' * 2)
    print("Data types:")
    print()
    print(df.dtypes)
    print('\n' * 2)
    print("Distribution of the variables in the dataframe:")
    print()
    print(df.describe())


def create_hist(df, column_name, x_lab, y_lab, title):
    '''
    '''
    df[column_name].hist(bins=50, grid=False, xlabelsize=12, ylabelsize=12)
    plt.xlabel(x_lab, fontsize=15)
    plt.ylabel(y_lab, fontsize=15)
    plt.show()


def create_boxplot(df, column_name, by_variable, x_lab, y_lab, title):
    '''
    This function will print a boxplot that shows the
    distribution of data
    Inputs:
        - df: a dataframe
        - column_name: string, the column to show distribution
        - by_variable: string, the column to create comparison groups
        - x_lab: string, the label for the x axis
        - y_lab: string, the label for the y axis
        - title: string, the title of the boxplot
    '''
    df.boxplot(column=column_name,
               by=by_variable).set(xlabel=x_lab,
                                   ylabel=y_lab)
    plt.title(title)
    plt.suptitle('')
    plt.show()


def find_outliers(df, column_name):
    '''
    This function will return a dataframe that contains
    the outliers for a given column
    Inputs:
        - df: a dataframe
        - column_name: string
    '''
    outliers = df[np.abs((df[column_name] -
                          df[column_name].mean()) / df[column_name].std()) > 3]
    print("Outliers' shape:")
    print(outliers.shape)
    return outliers


def create_corr_heatmap(df):
    '''
    This function will take in a dataframe and produce a
    correlation heatmap
    '''
    corr = df.corr()
    sns.heatmap(corr,
                xticklabels=corr.columns,
                yticklabels=corr.columns,
                cmap=sns.diverging_palette(220, 20, n=7, as_cmap=True))
    plt.show()


def pre_process(df, col_to_upcode=None):
    '''
    This function will fill in missing values and upcode
    values that seem illogical
    Inputs:
        - df: a dataframe
        - col_to_upcode: string, the column that needs to be upcoded
    '''
    print("Brief overview of the number of missing"
          "observations for each column:")
    print()
    print(df.isnull().sum().sort_values(ascending=False))
    print('\n'*2)
    null_col_list = df.columns[df.isna().any()].tolist()
    print()
    print("List of columns that contain missing data:")
    print()
    print(null_col_list)
    print('\n'*2)
    for col in null_col_list:
        try:
            df[col].fillna(df[col].median(), inplace=True)
        except:
            print("Can't fill missing values for non-numeric column {}".format(col))
            continue
    if col_to_upcode:
        df.loc[df[col_to_upcode] > 1, [col_to_upcode]] = 1
    print("Brief overview of the data distribution after pre-processing:")
    print()
    print(df.describe())


def convert_to_categorical_using_qcut(df, old_col, new_col, num_bin, label_list):
    '''
    This function converts a continuous variable into categorical
    using the pd.qcut method. The dataframe is transformed, function
    returns nothing.
    Inputs:
        - df: a dataframe
        - olc_col: (string) the name of the column to be converted
        - new_col: (string) the name of the new column that contains the
        categorical variable
        - num_bin: (int) number of bins
        - label_list: (list) list of labels for the categories
    '''
    df[new_col] = pd.qcut(df[old_col],
                          num_bin,
                          labels=label_list)


def convert_to_categorical_using_cut(df, old_col, new_col, bins, labels):
    '''
    This function converts a continuous variable into categorical
    using the pd.cut method. The dataframe is transformed, function
    returns nothing.
    Inputs:
        - df: a dataframe
        - olc_col: (string) the name of the column to be converted
        - new_col: (string) the name of the new column that contains the
        categorical variable
        - bins: (int) number of bins
        - labels: (list) list of labels for the categories
    '''
    df[new_col] = pd.cut(df[old_col],
                         bins=bins,
                         labels=labels,
                         include_lowest=True,
                         right=True)