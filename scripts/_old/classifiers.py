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
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_recall_curve
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier


def build_knn(y_train, x_train, k, weights='uniform', metric='minkowsky'):
    '''
    Build KNN classifier
    Inputs:
        y_train: Pandas Series
        x_train: Pandas DataFrame
        k: integer
        weights: str
        metric: str
    Output
        knn classifier
    '''
    knn = KNeighborsClassifier(n_neighbors=k, metric=metric, weights=weights)
    return knn.fit(x_train, y_train)


def build_tree(y_train, x_train, max_depth, criterion):
    '''
    Fit a decision tree model to the data provided, given the ycol and the set
    of x cols.
    Inputs:
        y_train: Pandas Series
        x_train: Pandas DataFrame
        max_depth: int
        criterion: str
    Output:
        Tree Classifier
    '''

    tree = DecisionTreeClassifier(max_depth=max_depth, criterion=criterion)
    return tree.fit(x_train, y_train)


def build_logistic_reg(y_train, x_train, C, penalty, fit_intercept, seed):
    '''
    Fit a logistic regression to the data provided, given the ycol and the set
    of x cols.
    Inputs:
        y_train: Pandas Series
        x_train: Pandas DataFrame
        C: float
        penalty: str
        seed: int
    Output:
        lreg Classifier
    '''
    lrf = LogisticRegression(random_state=seed, penalty=penalty, C=C,
                             fit_intercept=fit_intercept, solver='liblinear')
    return lrf.fit(x_train, y_train)


def build_svm(y_train, x_train, C, seed):
    '''
    Fit a SVM to the data provided, given the ycol and the set
    of x cols.
    Inputs:
        y_train: Pandas Series
        x_train: Pandas DataFrame
        C: float
        penalty: str
    Output:
        lreg Classifier
    '''
    svm = LinearSVC(random_state=seed, C=C)
    return svm.fit(x_train, y_train)


def build_random_forest(y_train, x_train, max_depth, criterion, n_estimators,
                        seed):
    '''
    Fit a random forest model to the data provided, given the ycol and the set
    of x cols.
    Inputs:
        y_train: Pandas Series
        x_train: Pandas DataFrame
        max_depth: int
        criterion: str
        n_estimators: int
    Output:
        RF Classifier
    '''
    forest = RandomForestClassifier(max_depth=max_depth, criterion=criterion,
                                    n_estimators=n_estimators, random_state=seed)
    return forest.fit(x_train, y_train)


def build_gradient_boost(y_train, x_train, max_depth, n_estimators,
                        loss, seed):
    '''
    Fit a random forest model to the data provided, given the ycol and the set
    of x cols.
    Inputs:
        y_train: Pandas Series
        x_train: Pandas DataFrame
        max_depth: int
        loss: str
        n_estimators: int
    Output:
        GB Classifier
    '''
    gb = GradientBoostingClassifier(
                max_depth=max_depth, n_estimators=n_estimators,
                loss=loss, random_state=seed)

    return gb.fit(x_train, y_train)


def predict_proba(classifier, x_test):
    '''
    Predict probability using classifier and text set
    Inputs:
        classifier: SKlearn classifier
        x_test: Pandas DataFrame
    Output:
        Pandas Series
    '''
    if isinstance(classifier, LinearSVC):
    	pred_scores = classifier.decision_function(x_test)
    	return pd.Series(pred_scores)
    pred_scores = classifier.predict_proba(x_test)
    return pd.Series([x[1] for x in pred_scores])


def classify(y_prob, threshold):
    '''
    Classify a Pandas Series given a threshold.
        y_prob: Pandas Series
        threshold: float
    Output:
        Pandas Series
    '''
    y_pred = np.where(y_prob >= threshold, 1, 0)
    return y_pred


def build_evaluation_metrics(y_true, y_pred, y_score=None, output_dict=True):
    '''
    Get evaluation metrics of Precision, Recall, F1
    Inputs:
        y_true: Pandas Series
        y_pred: Pandas Series
        y_score: Pandas Series
        Output_dict: bool
    Output:
        dict
    '''
    report_dict = classification_report(y_true, y_pred,
                                        output_dict = output_dict,)
    key_metrics = {'precision': report_dict['1']['precision'],
                   'recall':  report_dict['1']['recall'],
                   'f1': report_dict['1']['f1-score'],
                   'accuracy': accuracy_score(y_true, y_pred)}
    if not y_score is None:
        key_metrics['roc auc'] = roc_auc_score(y_true, y_score, average='micro')

    return key_metrics


def build_all_metrics_for_model(y_prob, y_test, threshold):
    '''
    Estimate y_pred and return evaluation metrics for a model.
    Inputs:
        model: Sk learn model
        y_test: PD Series
        x_test: PD Series
        threshold: float
    '''
    y_pred = classify(y_prob, threshold)
    metrics = build_evaluation_metrics(y_test, y_pred, y_prob)
    
    return metrics


def plot_precision_recall(y_true, y_score, x_axis='threshold'):
    '''
    Plot precision and recall curves. If x_axis == 'threshold',
    x axis is decision threshold, if x_axis is 'recall', x_axis
    is recall.

    Inputs:
        y_true: Pandas Series
        y_pred: Pandas Series
        x_axis: (threshold', 'recall') str
    Output:
        map
    '''
    precision, recall, threshold = precision_recall_curve(y_true, y_score)
    threshold = np.append(threshold, 1)

    if x_axis == 'threshold':
        plt.step(threshold, precision, color='b', alpha=0.4,
             where='pre', label='Precision')
        plt.step(threshold, recall, color='r', alpha=0.4,
             where='pre', label='Recall')
        plt.xlabel('Threshold')
        plt.ylabel('Value')
        plt.legend()
        plt.ylim([0.0, 1])
        plt.xlim([0.0, 1.0])
        plt.title('Precision-Recall curve by threshold: AP={0:0.2f}'
                    .format(precision.mean()))
    else:
        plt.step(recall, precision, color='b', alpha=0.4 ,
         where='post')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.ylim([0.0, 1])
        plt.xlim([0.0, 1.0])
        plt.title('2-class Precision-Recall curve: AP={0:0.2f}'
                    .format(precision.mean()))
    plt.show()
