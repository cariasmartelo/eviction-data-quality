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

def update_dict(results_dict, metrics, model_name, cross_k, threshold):
    '''
    Update results dictionary using metrics from model.
    Inputs:
        results_dict: dict
        model_name: name
        cross:k: int
        metrics: dict
    '''
    results_dict['model'].append(model_name)
    results_dict['cross_k'].append(cross_k)
    results_dict['threshold'].append(threshold)
    results_dict['precision'].append(metrics['precision'])
    results_dict['recall'].append(metrics['recall'])
    results_dict['AUC ROC'].append(metrics.get('roc auc', 0))

    return results_dict

def run_model(x_train, y_train, x_test, classif_model, model_params):
    '''
    Create classification model and return y_score. The function takes model,
    which is a function, and a dict with the parameters of that model.
    Inputs:
        x_train: Pandas Series
        y_train: Pandas Series
        x_test: Pandas Series
        classif_model: function
        model_params: dict
    '''
    model = classif_model(y_train, x_train, **model_params)
    y_score = classif.predict_proba(model, x_test)

    return y_score

def run_(x, y, semester, params):
    '''
    run cross validation and produce output df.
    Inputs:
        x: Pandas DataFrame
        y: Pandas Series
        semester: Pandas Series
        pamarams: dict
    Output:
        dict
    '''
    results = {'model': [],
               'cross_k': [],
               'threshold': [],
               'precision': [],
               'recall': [],
               'AUC ROC': []}

    models_funcitons = {
        'KNN': classif.build_knn,
        'decision_tree': classif.build_tree,
        'logistic_reg': classif.build_logistic_reg,
        'svm': classif.build_svm,
        'random_forest': classif.build_random_forest,
        'gradient_boost': classif.build_gradient_boost}

    cross_k = 0
    semesters = semester.nunique()
    tscv = TimeSeriesSplit(n_splits=semesters - 1)
    for train_index, test_index in tscv.split(range(semesters)):
        cross_k += 1
        print("Begining cross k: {}".format(cross_k))
        x_train = x[semester.isin(train_index)]
        y_train = y[semester.isin(train_index)]
        x_test = x[semester.isin(test_index)]
        y_test = y[semester.isin(test_index)]
        print("Train set has {} rows, with semester values of {}"
                .format(len(x_train), train_index))
        print("Test set has {} rows, with semester values of {}"
                .format(len(x_test), test_index))

        for model in params['models_to_run']:
            print()
            print('Fitting {}\n'.format(model))
            y_prob = run_model(x_train, y_train, x_test,
                models_funcitons[model], params[model])
            if model == 'svm':
                thresholds = params['svm_scores']
            else:
                thresholds = params['thresholds']

            for threshold in thresholds:
                print('Classifying model {} with threshold {}'
                      .format(model, threshold))
                metrics = classif.build_all_metrics_for_model(y_prob,
                                                              y_test,
                                                              threshold)
                results = update_dict(results, metrics, model, cross_k,
                                      threshold)
    return results
        

def run(x, y, semester, params):
    '''
    run cross validation and produce output df.
    Inputs:
        x: Pandas DataFrame
        y: Pandas Series
        semester: Pandas Series
        pamarams: dict
    Output:
        dict
    '''
    results = {'model': [],
               'cross_k': [],
               'threshold': [],
               'precision': [],
               'recall': [],
               'AUC ROC': []}

    models_funcitons = {
        'KNN': classif.build_knn,
        'decision_tree': classif.build_tree,
        'logistic_reg': classif.build_logistic_reg,
        'svm': classif.build_svm,
        'random_forest': classif.build_random_forest,
        'gradient_boost': classif.build_gradient_boost}

    cross_k = 0
    semesters = semester.nunique()
    tscv = TimeSeriesSplit(n_splits=semesters - 1)
    for train_index, test_index in tscv.split(range(semesters)):
        cross_k += 1
        print("Begining cross k: {}".format(cross_k))
        x_train = x[semester.isin(train_index)]
        y_train = y[semester.isin(train_index)]
        x_test = x[semester.isin(test_index)]
        y_test = y[semester.isin(test_index)]
        print("Train set has {} rows, with semester values of {}"
                .format(len(x_train), train_index))
        print("Test set has {} rows, with semester values of {}"
                .format(len(x_test), test_index))

        thresholds = params['thresholds']
        models_to_run = params['models_to_run']


        if 'KNN' in models_to_run:
            model = classif.build_knn(y_train, x_train,
                        k=params['KNN']['k'],
                        weights=params['KNN']['weights'],
                        metric=params['KNN']['metric'])
            y_prob = classif.predict_proba(model, x_test)
            for threshold in thresholds:
                metrics = classif.build_all_metrics_for_model(y_prob, y_test,
                                                              threshold)
                results = update_dict(results, metrics, 'KNN', cross_k, threshold)

        if 'DT' in models_to_run:
            model = classif.build_tree(y_train, x_train,
                        max_depth=params['decision_tree']['max_depth'],
                        criterion=params['decision_tree']['criterion'])
            y_prob = classif.predict_proba(model, x_test)
            for threshold in thresholds:
                metrics = classif.build_all_metrics_for_model(y_prob, y_test,
                                                              threshold)
                results = update_dict(results, metrics, 'DT', cross_k, threshold)

        if 'LR' in models_to_run:
            model = classif.build_logistic_reg(y_train, x_train,
                        C=params['logistic_reg']['C'],
                        penalty=params['logistic_reg']['penalties'],
                        fit_intercept=params['logistic_reg']['fit_intercept'],
                        seed = params['seed'])
            y_prob = classif.predict_proba(model, x_test)
            for threshold in thresholds:
                metrics = classif.build_all_metrics_for_model(y_prob, y_test,
                                                              threshold)
                results = update_dict(results, metrics, 'LR', cross_k, threshold)

        if 'SVM' in models_to_run:
            model = classif.build_svm(y_train, x_train,
                        C=params['svm']['C'],
                        seed = params['seed'])
            confidence_score = model.decision_function(x_test)
            for threshold in params['svm']['scores']:
                y_pred =  [1 if x > threshold else 0 for x in confidence_score]
                metrics = classif.build_evaluation_metrics(y_test, y_pred)
                results = update_dict(results, metrics, 'SVM', cross_k, threshold)


        if 'RF' in models_to_run:
            model = classif.build_random_forest(y_train, x_train,
                        max_depth=params['random_forest']['max_depth'],
                        criterion=params['random_forest']['criterion'],
                        n_estimators=params['random_forest']['n_estimators'],
                        seed=params['seed'])
            y_prob = classif.predict_proba(model, x_test)
            for threshold in thresholds:
                metrics = classif.build_all_metrics_for_model(y_prob, y_test,
                                                              threshold)
                results = update_dict(results, metrics, 'RF', cross_k, threshold)


        if 'GB' in models_to_run:
            model = classif.build_gradient_boost(
                        y_train, x_train,
                        max_depth=params['gradient_boost']['max_depth'],
                        n_estimators=params['gradient_boost']['n_estimators'],
                        loss=params['gradient_boost']['loss'],
                        seed=params['seed'])
            y_prob = classif.predict_proba(model, x_test)
            for threshold in thresholds:
                metrics = classif.build_all_metrics_for_model(y_prob, y_test,
                                                              threshold)
                results = update_dict(results, metrics, 'GB', cross_k, threshold)


    return results


