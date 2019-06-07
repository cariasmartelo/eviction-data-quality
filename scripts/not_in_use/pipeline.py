'''
CAPP 30254 1 Machine Learning for Public Policy
Pipeline functions for HW3
Spring 2019
Professor: Rayid Ghani
Camilo Arias
'''
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import tree
from shapely.geometry import Point
import geopandas as gpd
from datetime import datetime, timedelta

def load_from_csv(csv_file_path):
    '''
    Load csv file with project data to a Pandas DataFrame.
    Inputs:
        csv_file: Str
    Output:
        Pandas DataFrame
    '''
    dtypes = {'projectid': str,
              'teacher_acctid': str,
              'schoolid ': str,
              'school_ncesid': str,
              'school_latitude': float,
              'school_longitude': float,
              'school_city': str,
              'school_state': str,
              'school_metro': str,
              'school_district': str,
              'school_county': str,
              'school_charter': str,
              'school_magnet': str,
              'teacher_prefix': str,
              'primary_focus_subject': str,
              'primary_focus_are': str,
              'secondary_focus_subject': str,
              'secondary_focus_area': str,
              'resource_type': str,
              'poverty_level': str,
              'grade_level': str,
              'total_price_including_optional_support': float,
              'students_reached': float,
              'eligible_double_your_impact_match': str,
              'date_posted': str,
              'datefullyfunded': str}
    to_int = {'f': 0,
                  't': 1}

    projects_df = pd.read_csv(csv_file_path, dtype=dtypes)
    projects_df['date_posted'] = pd.to_datetime(projects_df['date_posted'],
                                                format='%m/%d/%y')
    projects_df['datefullyfunded'] = pd.to_datetime(projects_df['datefullyfunded'],
                                                    format='%m/%d/%y')
    for col in ['school_charter', 'school_magnet',\
                'eligible_double_your_impact_match']:
        projects_df[col] = projects_df[col].map(to_int)

    return projects_df


def create_outcome_var(projects_df, days):
    '''
    Create variable True if project was fully funded within days days.
    Inputs:
        projects_df: Pandas DataFrame
        days: int
    Output:
        Pandas DataFrame
    '''
    df = projects_df.loc[:]
    df['days_untill_funded'] = (df['datefullyfunded'] - df['date_posted']).dt.days
    outcome_var = "not_funded_in_{}_days".format(days)
    df[outcome_var] = np.where(df['days_untill_funded'] > days, 1, 0)

    return df.drop('days_untill_funded', axis=1)


def load_us_map(shapefile_path):
  '''
  load us map shapefile into geopandas DataFrame.
  Inputs:
    shalefile_path: str
  Output:
    Geopandas DataFrame
  '''
  us_gdf = gpd.read_file(shapefile_path)
  return us_gdf


def convert_to_geopandas(projects_df):
    '''
    Converts the pandas dataframe to geopandas to plot.
    Inputs:
        Pandas Df
    Output:
        Geopandas Df
    '''
    def shape_(x):

        '''
        Convert JSON location attribute to shapely.
        '''
        if isinstance(x, float):
            return np.NaN
        return shape(x)

    df = projects_df.loc[:]
    df['coordinates'] = list(zip(df['school_longitude'],
                                 df['school_latitude']))
    df['coordinates'] = df['coordinates'].apply(Point)
    geo_df = gpd.GeoDataFrame(df, crs = 'epsg:4326', geometry = df['coordinates'])

    return geo_df


def map_projects(projects_gdf, us_gdf, colorcol, dense=True, agg='mean'):
    '''
    Produce a map of the projects in US. If dense = True, the map will 
    be by states colored by agg function of colorcol. Else, the map will be
    a sparse map of the schools. If count, the map will be a dense map colored
    by frequency of schools in dataset. Cannot produce count sparce map.
        Inputs:
            projects_gdf: Geopandas DataFrame
            us_gdf: Geopandas DataGrame
            colorcol: str
            dense: bool
            agg: ('mean', 'meadian', 'max', 'mean', 'count') str
    '''
    if dense:
        if agg == 'count':
            project_by_state = (projects_gdf.groupby('school_state')
                                            .size().reset_index())
            project_by_state.rename(columns={0:'count'}, inplace = True)
            colorcol = 'count'

        else:
            project_by_state = (projects_gdf.groupby('school_state')
                                            .agg({colorcol: agg}))
    
        us_states_projs = us_gdf.merge(project_by_state, how='left',
                            left_on='STATE_ABBR', right_on = 'school_state')
        
    ax = us_gdf.plot(color="grey")
    if dense:
        us_states_projs.plot(ax=ax, column=colorcol, cmap='viridis',
                             legend=True)
        if agg == 'count':
            ax.set_title('Frequency of schools in US by State\n'
                         '(States without schools in grey)')
        else:
            ax.set_title('{} of {} in US by State\n(States without data'
                         ' in grey)'.format(agg.capitalize(),
                                            " ".join(colorcol.split("_"))
                                                .capitalize()))
        plt.show()

    else:
        projects_gdf.plot(ax=ax, column=colorcol, cmap='viridis',
                          legend=True, marker='.', markersize=2)
        ax.set_title('Project Schools in US by State by {} \n(States without data'
                         ' in grey)'.format(colorcol))
        plt.show()


def see_histograms(project_df, columns=None, restrict=None):
    '''
    Produce histograms of the numeric columns in project_df. If columns is
    specified, it produces histograms of those columns. If restrict dictionary
    is specified, restricts to the values inside the percentile range specified.
    Inputs:
        credit_df: Pandas DataFrame
        columns: [colname]
        restrict = dict
    Output:
        Individual Graphs (Num of graphs = Num of numeric cols)
    '''
    plt.clf()
    figs = {}
    axs = {}
    if not restrict:
        restrict = {}
    if not columns:
        columns = project_df.columns
    for column in columns:
        if not project_df[column].dtype.kind in 'ifbc':
            if project_df[column].nunique() <= 15:
                project_df[column].value_counts(sort=False).plot(kind='bar')
                plt.tight_layout()
                plt.title(column)
            continue
        if project_df[column].dtype.kind in 'c':
            project_df[column].value_counts().plot(kind='bar')

        if project_df[column].dtype.kind in 'c':

            continue
        if column in restrict:
            min_val = project_df[column].quantile(restrict[column][0])
            max_val = project_df[column].quantile(restrict[column][1])
            col_to_plot = (project_df.loc[(project_df[column] <= max_val)
                             & (project_df[column] >= min_val), column])
        else:
            col_to_plot = project_df[column]

        num_bins = min(20, col_to_plot.nunique())

        figs[column] = plt.figure()
        axs[column] = figs[column].add_subplot(111)
        n, bins, patches = axs[column].hist(col_to_plot, num_bins,
                                            facecolor='blue', alpha=0.5)
        axs[column].set_title(column)
    plt.show()


def summary_by_var(project_df, column, funct='mean', count=False):
    '''
    See data by binary column, aggregated by function.
    Input:
        credit_df: Pandas DataFrame
        funct: str
    Output:
        Pandas DF
    '''
    if not count:
        sum_table =  project_df.groupby(column).agg('mean').T
    else:
        sum_table = credit_df.groupby(column).size().T
    sum_table['perc diff'] = ((sum_table[1] / sum_table[0]) -1) * 100

    return sum_table


def make_dummies_from_categorical(project_df, columns=None, dummy_na=False):
    '''
    Function that takes a Pandas DataFrame and creates dummy variables for 
    the columns specified. If dummy_na True, make dummy for NA values.
        projec_df: Pandas DataFrame.
        columns: [str]
        dummy_na: bool
    Output:
        Pandas Data Frame.
    '''
    to_dummy = project_df.loc[:,project_df.nunique() < 100]

    dummies = pd.get_dummies(to_dummy, dummy_na=dummy_na, columns=columns)

    return dummies


def delete_columns(project_df, columns):
    '''
    Deletes columns of dataframe.
    Inputs:
        projec_df: Pandas DataFrame.
        columns: [str]
    Output:
        Pandas DataFrame
    '''
    df = project_df.loc[:]
    return df.drop(columns, axis=1)


def discretize(project_df, bins, equal_width=False, string=True):
    '''
    Function that takes a Pandas DataFrame and creates discrete variables for 
    the columns specified.
    Inputs:
        projects_df: Pandas DataFrame
        bins: int(num of bins)
        equal_width: bool
        string: book
    Output:
        Pandas Series.
    '''
    to_discretize = project_df.loc[:,(project_df.dtypes == 'int')\
                                    | (project_df.dtypes == 'float')]
    to_discretize = to_discretize.loc[:, to_discretize.nunique() > 30]

    if not equal_width:
        discretized =  to_discretize.apply(lambda x: pd.qcut(x, bins, labels=False))
    else:
        discretized =  to_discretize.apply(lambda x: pd.cut(x, bins, labels=False))
    if string:
        discretized = discretized.astype(str)

    return pd.concat([project_df, discretized.add_prefix('discr_')], axis=1)


def see_scatterplot(project_df, xcol, ycol, colorcol=None, logx=False,
                    logy=False, xjitter=False, yjitter=False):
    '''
    Print scatterplot of columns specified of the credit df. If color column
    is specified, the scatterplot will be colored by that column.
    Input:
        credit_df: Pandas DataFrame
        xcol: String
        ycol: String
        colorcol: String
        logx, logy: bool
        xiitter, yitter: bool
    Output:
        Graph
    '''
    df_to_plot = credit_df.loc[:]
    if xjitter:
        df_to_plot[xcol] = df_to_plot[xcol] +\
            np.random.uniform(-0.5, 0.5, len(df_to_plot[xcol]))\
            *df_to_plot[xcol].std()
    if yjitter:
        df_to_plot[ycol] = df_to_plot[ycol] +\
            np.random.uniform(-0.5, 0.5, len(df_to_plot[ycol]))\
            *df_to_plot[ycol].std()

    plt.clf()
    if not colorcol:
        df_to_plot.plot.scatter(x=xcol, y=ycol, legend=True, logx=logx,
                               logy=logy)
    else:
        df_to_plot.plot.scatter(x=xcol, y=ycol, c=colorcol, cmap='viridis',
                               legend=True, logx=logx, logy=logy)
    plt.title('Scatterplot of Credit DataFrame \n {} and {}'
                  .format(xcol, ycol))
    plt.show()


def group_by_days(date_series, days):
    '''
    Returns a pandas Series that indicates the semester of the date series
    as an integer from 0 to 3 (0 -> first semester of 2012, 3-> last semester
    of 2013)
    Inputs:
        date_series: Datetime series
    Output:
        Pandas Series
    '''
    start = date_series.min() + timedelta(days = days)
    end = date_series.max()
    bins = pd.date_range(start, end=end, freq='{}d'.format(days))

    def _group(x, bins):
        rv = 0
        for i, v in enumerate(bins):
            if x >= v:
                rv += 1
        return rv
    return (date_series.apply(_group, args=(bins,)), len(bins))

def get_all_combinations(model_dict):
    '''
    From a dictionary with all the models and all the parameters, 
    returns a dictionary of models mapped to a list of different
    specifications.
    Inputs:
        model_dict: dict
    Output:
        dict
    '''
    #setting up all combinations
    models = {}
    for classifier, parameters in model_dict.items():
        models[classifier] = []
        keyword_params = [param for param in parameters]
        options = [option for option in parameters.values()]
        num_of_params = len(options)
        combinations = []
        for option_1 in options[0]:
            if num_of_params < 2:
                combinations.append((option_1,))
                continue
            for option_2 in options[1]:
                if num_of_params < 3:
                    combinations.append((option_1, option_2))
                    continue
                for option_3 in options[2]:
                    if num_of_params < 4:
                        combinations.append((option_1, option_2, option_3))
                        continue
                    for option_4 in options[3]:
                        if num_of_params < 5:
                            combinations.append((option_1, option_2, option_3, option_4))
                            continue
        for combination in combinations:
            specification = {} 
            for i, keyword_param in enumerate(keyword_params):
                specification[keyword_param] = combination[i]
            models[classifier].append(specification)
    return models

