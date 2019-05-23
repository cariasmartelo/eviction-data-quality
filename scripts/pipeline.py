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
from datetime import datetime


def create_outcome_var(eviction_df, days):
    '''
    Create variable True if project was fully funded within days days.
    Inputs:
        eviction_df: Pandas DataFrame
        days: int
    Output:
        Pandas DataFrame
    '''
    df = eviction_df.loc[:]
    df['days_untill_funded'] = (df['datefullyfunded'] - df['date_posted']).dt.days
    outcome_var = "not_funded_in_{}_days".format(days)
    df[outcome_var] = np.where(df['days_untill_funded'] > days, 1, 0)

    return df


def convert_to_geopandas(eviction_df):
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

    df = eviction_df.loc[:]
    df['coordinates'] = list(zip(df['school_longitude'],
                                 df['school_latitude']))
    df['coordinates'] = df['coordinates'].apply(Point)
    geo_df = gpd.GeoDataFrame(df, crs = 'epsg:4326', geometry = df['coordinates'])

    return geo_df


def see_histograms(eviction_df, columns=None, restrict=None):
    '''
    Produce histograms of the numeric columns in eviction_df. If columns is
    specified, it produces histograms of those columns. If restrict dictionary
    is specified, restricts to the values inside the percentile range specified.
    Inputs:
        eviction_df: Pandas DataFrame
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
        columns = eviction_df.columns
    for column in columns:
        print(column)
        if not eviction_df[column].dtype.kind in 'ifbc':
            try:
                if eviction_df[column].nunique() <= 15:
                    eviction_df[column].value_counts(sort=False).plot(kind='bar')
                    plt.tight_layout()
                    plt.title(column)
                continue
            except: continue
        if eviction_df[column].dtype.kind in 'c':
            eviction_df[column].value_counts().plot(kind='bar')

        if eviction_df[column].dtype.kind in 'c':

            continue
        if column in restrict:
            min_val = eviction_df[column].quantile(restrict[column][0])
            max_val = eviction_df[column].quantile(restrict[column][1])
            col_to_plot = (eviction_df.loc[(eviction_df[column] <= max_val)
                             & (eviction_df[column] >= min_val), column])
        else:
            col_to_plot = eviction_df[column]

        num_bins = min(20, col_to_plot.nunique())

        figs[column] = plt.figure()
        axs[column] = figs[column].add_subplot(111)
        n, bins, patches = axs[column].hist(col_to_plot, num_bins,
                                            facecolor='blue', alpha=0.5)
        axs[column].set_title(column)
    plt.show()


def summary_by_var(eviction_df, column, funct='mean', count=False):
    '''
    See data by binary column, aggregated by function.
    Input:
        eviction_df: Pandas DataFrame
        funct: str
    Output:
        Pandas DF
    '''
    if not count:
        sum_table =  eviction_df.groupby(column).agg('mean').T
    else:
        sum_table = eviction_df.groupby(column).size().T
    #sum_table['perc diff'] = ((sum_table[1] / sum_table[0]) -1) * 100

    return sum_table


def make_dummies_from_categorical(eviction_df, columns, dummy_na=False):
    '''
    Function that takes a Pandas DataFrame and creates dummy variables for 
    the columns specified. If dummy_na True, make dummy for NA values.
        eviction_df: Pandas DataFrame.
        columns: [str]
        dummy_na: bool
    Output:
        Pandas Data Frame.
    '''
    #df = eviction_df.loc[:]
    #for col in columns:
    dummies = pd.get_dummies(eviction_df, dummy_na=dummy_na, columns= columns)
    #    df = df.join(dummies)

    return dummies


def delete_columns(eviction_df, columns):
    '''
    Deletes columns of dataframe.
    Inputs:
        eviction_df: Pandas DataFrame.
        columns: [str]
    Output:
        Pandas DataFrame
    '''
    df = eviction_df.loc[:]
    return df.drop(columns, axis=1)


def discretize(serie, bins, equal_width=False, string=False):
    '''
    Function to discretize a pandas series based on number of bins and
    wether bins are equal width, or have equal number of observations.
    If string = True, returns string type.
    Inputs:
        serie: Pandas Series
        bins: int(num of bins)
        equal_width: bool
    Output:
        Pandas Series.
    '''
    if not equal_width:
        return_serie =  pd.qcut(serie, bins)
    else:
        return_serie = pd.cut(serie, bins)
    if string:
        return_serie = return_serie.astype(str)

    return return_serie


def see_scatterplot(eviction_df, xcol, ycol, colorcol=None, logx=False,
                    logy=False, xjitter=False, yjitter=False):
    '''
    Print scatterplot of columns specified of the eviction_df. If color column
    is specified, the scatterplot will be colored by that column.
    Input:
        eviction_df: Pandas DataFrame
        xcol: String
        ycol: String
        colorcol: String
        logx, logy: bool
        xiitter, yitter: bool
    Output:
        Graph
    '''
    df_to_plot = eviction_df.loc[:]
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
    plt.title('Scatterplot of Eviction DataFrame \n {} and {}'
                  .format(xcol, ycol))
    plt.show()


def plot_map(eviction_gdf, variable, year, geography_name):
    '''
    Map by zip code the value of the column indicated in colorcol and the year.
    Inputs:
        eviction_gdf: GeoDataFrame
        variable: Str
        year: int
        geography_name: str
    Output:
        Map
    '''
    col_dict = {
        'n': 'name', 'pl': 'parent-location', 'p': 'population',
        'pro': 'pct-renter-occupied', 'mgr': 'median-gross-rent',
        'mhi': 'median-household-income', 'mpv': 'median-property-value',
        'rb': 'rent-burden', 'roh': 'renter-occupied-households',
        'pr': 'poverty-rate', 'pw': 'pct-white', 'paa': 'pct-af-am',
        'ph': 'pct-hispanic', 'pai': 'pct-am-ind', 'pa': 'pct-asian',
        'pnp': 'pct-nh-pi', 'pm': 'pct-multiple', 'po': 'pct-other',
        'e': 'evictions', 'ef': 'eviction-filings', 'er': 'eviction-rate',
        'efr': 'eviction-filing-rate', 'lf': 'low-flag'}

    colorcol = {v: i for i, v in col_dict.items()}[variable]
    colorcol += '-' + str(year)[-2:] #Use variable and year to get column name

    fig, ax = plt.subplots(figsize=(8, 12))
    eviction_gdf.plot(color="grey", ax=ax, edgecolor="black")
    eviction_gdf[eviction_gdf[colorcol].notna()].plot(ax=ax, column=colorcol,
                                                      cmap='viridis',
                                                      scheme='quantiles',
                                                      legend=True)

    ax.set_title('Tracts of {} by {} in {}\n(Tracts without data'
                 ' in grey)'.format(geography_name, " ".join(variable.split("-")),
                                    year))
    plt.show()


def set_semester(date_series):
    '''
    Returns a pandas Series that indicates the semester of the date series
    as an integer from 0 to 3 (0 -> first semester of 2012, 3-> last semester
    of 2013)
    Inputs:
        date_series: Datetime series
    '''
    bins = pd.date_range('2012-07-01', end='2014-1-31', freq='183d')

    def _semester(x, bins):
        rv = 0
        for i, v in enumerate(bins):
            if x >= v:
                rv += 1
        return rv
    return (date_series.apply(_semester, args=(bins,)), len(bins))
