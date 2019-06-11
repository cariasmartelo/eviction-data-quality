'''
Pipeline Codes for Eviction Project
Group 2
'''


from __future__ import division
import pandas as pd
import numpy as np
import os
import matplotlib
import matplotlib.pyplot as plt
import pylab as pl
from datetime import timedelta
from datetime import datetime
import random
from scipy import optimize
import time
import seaborn as sns
import csv
import math
from sklearn.model_selection import ParameterGrid, train_test_split
from sklearn.metrics import roc_auc_score 
from datetime import date, datetime, timedelta
from dateutil.relativedelta import relativedelta

from helper import *


RANDOM_STATE = 100


def get_threshold(num_array, quantile):
    '''
    This function finds the cut-off value for top k%
    most at risk census tracts (highest eviction rate)
    To get the top 10% eviction rate, use quantile = .9
    '''
    threshold = num_array.quantile(quantile)
    return threshold


def create_label(df, year_col, label_col, quantile, prediction_window):
    '''
    '''
    df = df.iloc[:,:]
    df['next_year'] = df['year'] + 1 
    small_df = df[['year', 'tract', 'eviction_filings_rate']]
    small_df.columns = ['next_year', 'tract', 'eviction_filings_rate_next_year']
    #create a master df, adding eviction filings rate next year to each row
    master_df = pd.merge(left=df, right=small_df, how='left', on=['next_year', 'tract'])

    years = master_df[year_col].unique().tolist()
    for year in years:
        threshold = get_threshold(master_df.loc[master_df[year_col] == (year + 1), label_col], quantile)
        master_df.loc[master_df['year'] == year, 'label'] = np.where(
            master_df.loc[master_df['year'] == year, 'eviction_filings_rate_next_year'] >= threshold, 1, 0)
        prev_threshold  = get_threshold(master_df.loc[master_df[year_col] == (year), label_col], quantile)
        master_df.loc[master_df['year'] == year, 'label_prev_year'] = np.where(
            master_df.loc[master_df['year'] == year, label_col] >= prev_threshold, 1, 0)

    return master_df


def find_nuls(df):
    '''
    This function finds all the null columns in the dataframe
    and return a list of such columns
    '''
    print(df.isnull().sum().sort_values(ascending=False))
    null_col_list = df.columns[df.isna().any()].tolist()
    return null_col_list


def fill_null_cols(df, null_col_list):
    '''
    '''
    for col in null_col_list:
        try:
            df[col].fillna(df[col].median(), inplace=True)
        except:
            print("Can't fill missing values for non-numeric column {}".format(col))
            continue


# def discretize_cols(df, old_col, num_bins=3, cats=False):
def discretize_cols(serie, num_bins=3, cats=False):
    '''
    This function converts a list of continous columns into categorical
    Inputs:
        - dataframe (pandas dataframe)
        - old col (string): label of column to discretize
        - num_bins (int): number of bins to discretize into
        - cats: a list of categories to organize the continuous values into
    Returns a pandas dataframe
    '''
    # new_col = old_col + '_group'
    # df[new_col] = pd.cut(df[old_col], 
    #                      bins=num_bins, 
    #                      labels=cats, 
    #                      right=True, 
    #                      include_lowest=True)
    # return df
    rv = pd.cut(serie, 
                bins=num_bins, 
                labels=cats, 
                right=True, 
                include_lowest=True)
    return rv


def convert_to_binary(df, cols_to_transform):
    '''
    This function converts a list of categorical columns into binary
    Inputs:
        - df (dataframe)
        - cols_to_transform (list)
    '''
    df = pd.get_dummies(df, dummy_na=True, columns=cols_to_transform)
    return df


def convert_to_datetime(df, cols_to_transform):
    '''
    This function converts a list of columns into datetime type
    '''
    for col in cols_to_transform:
        df[col] = pd.to_datetime(df[col])


def process_df(df, cols_to_discretize, num_bins, cats, cols_to_binary):
    '''
    This function puts together all the processing steps necessary for
    a dataframe
    '''
    fill_null_cols(df, find_nuls(df))
    # for col in cols_to_discretize:
    #     processed_df = discretize_cols(df, col, num_bins, cats)
    discrete = df[cols_to_discretize].apply(discretize_cols, args = (num_bins, cats))
    discrete = discrete.add_suffix('_group')
    processed_df = pd.concat([df, discrete], axis=1)
    processed_df = convert_to_binary(processed_df, cols_to_binary)
    return processed_df


def process_train_data(rv, cols_to_discretize, num_bins,
                       cats, cols_to_binary):
    '''
    This function will consider the train and test set separately 
    and perform processing functions on each set
    '''
    processed_rv = {}
    for split_date, data in rv.items():
        train = data[0]
        test = data[1]
        #process train & test set (fill in nulls, discretize, convert to binary)
        processed_train = process_df(train, cols_to_discretize, 
                                     num_bins, cats, cols_to_binary)
        processed_test = process_df(test, cols_to_discretize, 
                                    num_bins, cats, cols_to_binary)

        processed_rv[split_date] = [processed_train, processed_test]
    return processed_rv


def clf_loop_cross_validation(models_to_run, clfs, grid, processed_rv, 
                              predictors, outcome, thresholds, time_col, 
                              bias_lst):
    '''
    This function will produce a dataframe to store the performance metrics
    of all the models created.
    Inputs:
        - models_to_run: a list of models to run 
        - clfs: a dictionary with all the possible classifiers
        - grid: a dictionary that documents all the variation of parameters
        for each classifier
        - processed_rv: a dictionary that maps a split date to a list that
        contains the processed train set and processed test set for that
        particular split date
        - predictors: the list of features
        - outcome: the label column
        - thresholds: the thresholds of interest that we will use to build
        performance metrics
        - time_col: the date columns (will be used to check start and end
        date of train and test set)
    Returns:
        - a dataframe of results
    '''
    metrics = ['p_at_', 'recall_at_', 'f1_at_']
    metric_cols = []
    for thres in thresholds:
        for metric in metrics:
            metric_cols.append(metric + str(thres)) #cycling through all metrics and create column labels
    COLS = ['model_type', 'clf', 'parameters', 'split_date'] + \
           ['min_year_in_train', 'max_year_in_train', 'min_year_in_test', 'max_year_in_test'] +\
           ['baseline'] + \
           metric_cols + \
           ['auc-roc']
    
    results_df =  pd.DataFrame(columns=COLS)
    i = 0
    for n in range(1, 2):
        for split_date, data in processed_rv.items():
            train_set, test_set, bias_set = data
            #Extract features and labels for train set and test set
            X_train = train_set[predictors]
            X_test = test_set[predictors]
            y_train = train_set[outcome]
            y_test = test_set[outcome]
            #Calculate train start/end date and test start/end date
            train_start = train_set[time_col].min()
            train_end = train_set[time_col].max()
            test_start = test_set[time_col].min()
            test_end = test_set[time_col].max()

            for index, clf in enumerate([clfs[x] for x in models_to_run]):
                model_name = models_to_run[index]
                print(model_name)
                parameter_values = grid[models_to_run[index]]
                #for each classifier, fit the model based on the train set
                for p in ParameterGrid(parameter_values):
                    try:
                        clf.set_params(**p) 
                        clf.fit(X_train, y_train)
                        if model_name == 'SVM':
                            y_pred_probs = clf.decision_function(X_test)
                        else:
                            y_pred_probs = clf.predict_proba(X_test)[:,1]                        
                        y_pred_probs_sorted, y_test_sorted = zip(*sorted(zip(
                            y_pred_probs, y_test), reverse=True))

                        #apply the model created to the test set
                        #calculate the metrics (precision, recall, f1) based on the thresholds
                        metrics_stats = []
                        for thres in thresholds:
                            pres = precision_at_k(y_test_sorted, y_pred_probs_sorted, thres)
                            rec = recall_at_k(y_test_sorted, y_pred_probs_sorted, thres)
                            f1 = f1_at_k(y_test_sorted, y_pred_probs_sorted, thres)
                            metrics_stats.extend([pres, rec, f1])
                        #for each model, store all relevant information in a list
                        #this list will later be fed into the outcome dataframe as a row
                        #the value in the list called row correspond to the columns created in COLS (line 154)
                        row = [models_to_run[index], clf, p, split_date] + \
                              [train_start, train_end, test_start, test_end] + \
                              [precision_at_k(y_test_sorted, y_pred_probs_sorted, 100)] + \
                              metrics_stats + \
                              [roc_auc_score(y_test, y_pred_probs)]
                        #insert row into the outcome dataframe
                        results_df.loc[len(results_df)] = row
                        # calculate bias
                        test_set['score'] = y_pred_probs
                        for bias_col in bias_lst:
                            test_set[bias_col] = bias_set[bias_col]
                        bias_df = test_set[[score, outcome] + bias_lst]
                        model_id = [i for i in range(len(bias_df))]
                        bias_df['entity_id'] = model_id
                        bias_df.rename({outcome:label_value}, axis='columns')
                        assess_bias(bias_df, 
                                    metrics = ['ppr','pprev','fnr','fpr', 'for'], 
                                    min_group_size = None)
                        i +=1
                        #Plot the precision recall curves
                        plot_precision_recall_n(y_test, y_pred_probs, clf)
                    except IndexError as e:
                        print('Error:',e)
                        continue
    return results_df


def temporal_validation(df, date_col, prediction_window, start_time, end_time, 
                        len_train, bias_lst, gap=0):
    '''
    Create a dictionary that maps a key that is the validation date with a list
    of train set and test set that correspond to that validation date.
    Train set will contain records before validation date
    Test set will contain records after validation date
    Inputs:
        - df: a dataframe
        - date_col: the date column
        - prediction_windows: a list that contains all prediction windows in months
        - gap: the number of days between train end date and test start date (not using for now)
        - start_time: string, earliest datetime in the data
        - end_time: string, latest datetime in the data
        - len_train: the length of the train set
        - bias_lst: the list of columns we are calculating bias scores on
    Outputs:
        a dictionary that maps the validation date to a list that contains the
        corresponding train set, test set and bias set for that date
    '''
    start_time_date = datetime.strptime(start_time, '%Y-%m-%d')
    end_time_date = datetime.strptime(end_time, '%Y-%m-%d')
    train_start_time = start_time_date
    rv = {}
    while train_start_time <= end_time_date - relativedelta(months=prediction_window):
        train_end_time = train_start_time + relativedelta(months=len_train)    
        test_start_time = train_end_time 
        test_end_time = test_start_time + relativedelta(months=prediction_window)
        train_set = df[(df[date_col] >= train_start_time) &
                           (df[date_col] < train_end_time)]
        test_set = df[(df[date_col] >= test_start_time) &
                           (df[date_col] < test_end_time)]
        bias_set = test_set[bias_lst]
        rv[test_start_time] = [train_set, test_set, bias_set]
        #once done, increment len train by 12 months
        len_train += 12
    return rv


def get_continuous_variables(df, nunique=30):
    '''
    Get the colname of the continuous functions that have more than nunique
    unique values.
    Inputs:
        df DataFrame
        nunique: int
    Output:
        list of colnames
    '''
    to_discretize = df.loc[:,(df.dtypes == 'int')\
                           | (df.dtypes == 'float')]
    to_discretize = to_discretize.loc[:, to_discretize.nunique() > nunique]
    cols_to_discretize = list(to_discretize.columns)
    return list(cols_to_discretize)


def assess_bias(bias_df, metrics = ['fnr', 'for'], min_group_size = None):
    '''
    This function creates bar charts for bias metrics given.
    bias_df = dataframe with ID, label, predicted scores already taking into account the population threshold, and 
    '''
    g = Group()
    xtab, _ = g.get_crosstabs(bias_df)
    aqp = Plot()
    p = aqp.plot_group_metric_all(xtab, 
                                  metrics=metrics, 
                                  ncols=len(metrics), 
                                  min_group_size = min_group_size)
    p.show()
    return

