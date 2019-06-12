import download
import mlpipeline
import helper
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt


params = {
    'to_exclude': ['year', 'label'],
    'top_x_percent': 0.1,
    'date_col': 'filing_year',
    'prediction_window': 12,
    'start_time': '2010-01-01',
    'end_time': '2016-01-01',
    'len_train': 36,
    'discrete_bins': 3,
    'cats': ['low', 'medium', 'high'],
    'outcome': 'label',
    'model_params': {
    'LR': {
        'C': 10,
        'class_weight': None,
        'penalty': 'l2',
        'fit_intercept': True,
        'intercept_scaling': 1,
        'max_iter': 100,
        'multi_class': 'warn',
        'n_jobs': None,
        'random_state': 1234,
        'solver': 'warn',
        'tol': 0.0001,
        'verbose': 0,
        'warm_start': False}},
    'RF': {
        
    }
        }


eviction = download.load_evict('../inputs/eviction_data_tract.csv')
crime = download.load_crime('../inputs/crime_by_tract.csv')
buildings = download.load_building('../inputs/building_violation_by_tract.csv')
acs = download.load_acs('../inputs/acs_year_tract.csv')
education = download.load_education('../inputs/educ_year_tract.csv')
tracts = download.load_tract('../inputs/ch_opdat/tracts.csv')

eviction_df = mlpipeline.create_label(eviction, 'year', 'eviction_filings_rate', 1 - params['top_x_percent'],
                                      params['prediction_window'])
eviction_df = eviction_df.drop(['eviction_filings_rate_next_year', 'next_year'], axis = 1)

train_set = eviction_df.loc[eviction_df['year'] != 2017].copy()
test_set = eviction_df.loc[eviction_df['year'] == 2017].copy()

cols_to_discretize = mlpipeline.get_continuous_variables(eviction_df)
cols_to_binary = []
for col in cols_to_discretize:
    cols_to_binary.append(col + "_group")

process_train = mlpipeline.process_df(train_set, cols_to_discretize, params['discrete_bins'],
                                       params['cats'], cols_to_binary)

selected_features = list(process_train.loc[:,process_train.apply(lambda x: x.isin([0, 1]).all())].columns)
selected_features.remove('label')
predictors = selected_features

process_test = mlpipeline.process_df(test_set, cols_to_discretize, params['discrete_bins'],
                                      params['cats'], cols_to_binary)

x_train = process_train[selected_features]
y_train = process_train['label']
x_test = process_test[selected_features]

clf_lr = LogisticRegression(C= 10,
                        class_weight= None,
                        penalty= 'l2',
                        fit_intercept= True,
                        intercept_scaling= 1,
                        max_iter= 100,
                        multi_class= 'warn',
                        n_jobs= None,
                        random_state= 1234,
                        solver= 'warn',
                        tol= 0.0001,
                        verbose= 0,
                        warm_start= False)
clf_lr.fit(x_train, y_train)
y_pred_probs = clf_lr.predict_proba(x_test)
test_set['predicted_lr_score'] = y_pred_probs[:,1]
test_set.sort_values('predicted_lr_score', inplace=True, ascending=False)
test_set['prediction_lr'] = helper.generate_binary_at_k(test_set['predicted_lr_score'], 10)

clf_rf = RandomForestClassifier(bootstrap=True,
                             class_weight=None,
                             criterion='gini',
                             max_depth=50,
                             max_features='log2',
                             max_leaf_nodes=None,
                             min_impurity_decrease=0.0,
                             min_impurity_split=None,
                             min_samples_leaf=1,
                             min_samples_split=10,
                             min_weight_fraction_leaf=0.0,
                             n_estimators=10000,
                             n_jobs=-1,
                             oob_score=False,
                             random_state=None,
                             verbose=0,
                             warm_start=False)

clf_rf.fit(x_train, y_train)
y_pred_probs = clf_rf.predict_proba(x_test)
test_set['predicted_rf_score'] = y_pred_probs[:,1]
test_set.sort_values('predicted_rf_score', inplace=True, ascending=False)
test_set['prediction_rf'] = helper.generate_binary_at_k(test_set['predicted_rf_score'], 10)

test_set[['tract', 'year', 'prediction_lr', 'prediction_rf']].to_csv('../results/prediction_results.csv')

test_set['predicted_lr_score'].hist()
plt.title('LR predited histograms')
plt.show('../figures/y_pred_hist_lr.png')
mlpipeline.get_feature_importance('LR', x_train, clf_lr, n_importances=10)
predicted_tracts = test_set.loc[test_set['prediction_lr'] == 1, 'tract'].copy()

eviction_df['correct_with_baseline'] = eviction_df['label'] * eviction_df['label_prev_year']
(eviction_df.groupby('filing_year')['correct_with_baseline'].sum()/(int(tracts.shape[0]*0.1)))\
.to_csv('../results/baseline.csv', header=True)


