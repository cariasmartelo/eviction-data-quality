'''
CAPP 30254 1 Machine Learning for Public Policy
Classifier functions for HW3
Spring 2019
Professor: Rayid Ghani
Camilo Arias
'''
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn.tree as tree
import pipeline as ppln
import classifiers as classif
from sklearn.model_selection import TimeSeriesSplit

def update_dict(results_dict, metrics, model_name, cross_k, top_k,
                specification):
    '''
    THis function helps the run function update the dictionary with the
    results of the model. It takes a dictionary of the results so far and
    a dictionary of the metrics and updates the former one.
    Inputs:
        results_dict: dict
        model_name: name
        cross_k: int
        metrics: dict
        top_k: float
        specification: dict
    '''
    results_dict['model'].append(model_name)
    results_dict['parameters'].append(str(specification))
    results_dict['cross_k'].append(cross_k)
    results_dict['top_k'].append(top_k)
    results_dict['precision'].append(metrics['precision'])
    results_dict['recall'].append(metrics['recall'])
    results_dict['AUC ROC'].append(metrics.get('roc auc', 0))

    return results_dict

def get_iter_train_test(groups_serie, test_size, wait_size, num_of_trains):
    '''
    From a categorical Pandas Series that indicates the teporal classification
    of each observation, construct a list of lost of  train and test indexes to
    create the temporal holdouts. It takes into consideration the time
    that needs to be kept between training and testing. Test size and wait size
    indicate the number of temporal units of each of those. For example, if groups
    indicate bimesters, test_size of two would indicate that the testing is foo periods
    of 4 months. In this prediction I mapped everything to bimesters, so the test size
    is 3 (3 bimesters) and the wait_size if of 1 (1 bimester)
    Inputs:
        groups_serie: Pandas Series
        test_size: int
        wait_size: int
        num_of_trains: int
    '''
    groups = [i for i in range(groups_serie.nunique())]
    train_indexes = [[i for i in range(test_size)]]
    for i in range(num_of_trains -1):
        new_train = [i + test_size for i in train_indexes[-1][-test_size:]]
        train_indexes.append(train_indexes[-1] + new_train)
    wait_groups = [i for i in range(wait_size)]
    wait_index = [[w + tr[-1] + 1] for w in wait_groups for tr in train_indexes]
    test_groups = [i for i in range(test_size)]
    test_indexes = [[t + w[-1] + 1 for t in test_groups] for w in wait_index]

    return (train_indexes, test_indexes)


def run(x, y, groups_serie, test_size, wait_size, num_of_trains, models_dict,
        seed, top_ks, n_bins):
    '''
    run cross validation and produce output dict. The function takes the x and y
    DataFrames, a Pandas Series indicating the time group each observation is part
    of, a dictionary that has the models to run and the parameters of each, the
    size of the test in terms of groups and the size of the wait in terms of groups.
    It takes a random seed and the number of bins to discretize continuous variables.
    Inputs:
        x: Pandas DataFrame
        y: Pandas Series
        groups: Pandas Series
        models_dict: dict
        testsize: Number of groups in test
        wait_size: Number of groups in wait
        num_of_trains: int
    Output:
        dict
    '''

    #Setting initial model
    results = {'model': [],
               'parameters': [],
               'cross_k': [],
               'top_k': [],
               'precision': [],
               'recall': [],
               'AUC ROC': []}

    cross_k = 0

    ##Creating the list of train and test indexes that will help make the temporal
    ##holdouts.
    train_indexes, test_indexes = get_iter_train_test(groups_serie, test_size,
                                                      wait_size, num_of_trains)
    for i, train_index in enumerate(train_indexes):
        cross_k += 1
        print("Begining cross k: {}".format(cross_k))
        x_train = x[groups_serie.isin(train_index)]
        y_train = y[groups_serie.isin(train_index)]
        x_test = x[groups_serie.isin(test_indexes[i])]
        y_test = y[groups_serie.isin(test_indexes[i])]
        x_train = ppln.discretize(x_train, n_bins)

        print("Train set has {} rows, with group values of {}"
                .format(len(x_train), train_index))
        print("Test set has {} rows, with group values of {}"
                .format(len(x_test), test_indexes))

        ##Creating dummy variables for the train and the test set and 
        ##keeping only those.
        x_train = ppln.make_dummies_from_categorical(x_train)
        x_train = x_train.loc[:,x_train.dtypes == 'uint8']
        x_discrete = ppln.discretize(x, n_bins)
        x_test = ppln.make_dummies_from_categorical(x_discrete).loc[x_test.index,]
        x_test = x_test.loc[:, x_train.columns]


        for model, specifications in models_dict.items():
            print()
            print('Fitting {}\n'.format(model))
            for specification in specifications:
                y_prob = classif.run_model(x_train, y_train, x_test,
                					model, specification, seed)
                print('Built model {} with specification {}'
                          .format(model, specification))

                for top_k in top_ks:
                    metrics = classif.build_all_metrics_for_model(y_prob, y_test,
                                                                  top_k)
                    results = update_dict(results, metrics, model, cross_k,
                                          top_k, specification)
    return results
