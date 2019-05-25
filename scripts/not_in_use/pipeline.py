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

CHICAGO_LIMITS = {'lat': (41, 43),
                  'lon': (-88, -87)}



def import_eviction_csv(csv_file):
    '''
    Import eviction data from csv file.
    Inputs:
        csv_file: String
    Output:
        Pandas DataFrame
    '''
    eviction_lab_dtypes = {
        'GEOID': int, 'year': int, 'name': str, 'parent-location': str,
        'population': float, 'poverty-rate': float,
        'renter-occupied-households': float, 'pct-renter-occupied': float,
        'median-gross-rent': float, 'median-household-income': float,
        'median-property-value': float, 'rent-burden' : float,
        'pct-white': float, 'pct-af-am': float, 'pct-hispanic': float,
        'pct-am-ind': float, 'pct-asian': float, 'pct-nh-pi': float,
        'pct-multiple': float, 'pct-other': float, 'eviction-filings': float,
        'evictions': float, 'eviction-rate': float, 'eviction-filing-rate': float,
        'low-flag': float, 'imputed': float, 'subbed': float}

    eviction_df = pd.read_csv(csv_file, dtype=eviction_lab_dtypes)

    return eviction_df

def import_eviction_geojson(gpd_file):
    '''
    Import eviction data from geojson file.
    Inputs:
        gpd_file: String
    Output:
        GPD DataFrame
    '''
    eviction_gdf = gpd.read_file(gpd_file)

    return eviction_gdf



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


def convert_to_geopandas(df, x_col, y_col):
    '''
    Converts the pandas dataframe to geopandas to plot.
    Inputs:
        df Pandas Df
        coordinates_col str

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

    df = df.loc[:]
    df['coordinates'] = list(zip(df[x_col],
                                 df[y_col]))
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


def map_crimes_loc(geo_tract, geo_crime, filter_col=None, filter_vals=None, color_col=None):
    '''
    Produce a map of the crimes location in Chicago over a tract map.
    Inputs:
        geo_comm: GeoPandasDf
        geo_crime: GeoPandasDf
        filter_col: Str
        filter_vals: [str]
        color_col: str
    '''
    plt.clf()
    geo_crime = geo_crime[(geo_crime['latitude'].between(CHICAGO_LIMITS['lat'][0],
                                                        CHICAGO_LIMITS['lat'][1]))
                        & (geo_crime['longitude'].between(CHICAGO_LIMITS['lon'][0],
                                                          CHICAGO_LIMITS['lon'][1]))]
    plt.clf()
    base = geo_tract.plot(color='white', edgecolor='black')
    if filter_col:
        if not color_col:
            geo_crime[geo_crime[filter_col] == filter_vals].plot(ax=base, marker='.',
                color = 'red', markersize=4, legend = True)
        else:
            geo_crime[geo_crime[filter_col] == filter_vals].plot(ax=base, marker='.',
                column = color_col, markersize=4, legend = True)
    else:
        if not color_col:
            geo_crime.plot(ax=base, marker='.',
                color = 'red', markersize=4, legend = True)
        else:
            geo_crime.plot(ax=base, marker='.',
                column = color_col, markersize=4, legend = True)
    if not filter_col:
        plt.title('Crimes in Chicago (2017 and 2018)')
    else:
        plt.title('{}s in Chicago (2017 and 2018)'
                  .format(filter_vals.capitalize()))
    plt.show()
    plt.clf()


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



def convert_to_geopandas(df):
    '''
    Converts the pandas dataframe to geopandas DataFrame
    Inputs:
        df: Pandas DataFrame
    Output:
        Geopandas DataFrame
    '''
    def shape_(x):

        '''
        Convert JSON location attribute to shapely.
        '''
        if isinstance(x, float):
            return np.NaN
        return shape(x)


    df['geometry'] = df.location.apply(shape_)
    geo_df = gpd.GeoDataFrame(df, crs = 'epsg:4326', geometry = df['geometry'])

    return geo_df

def make_cross_var_year(df1, df2, df3, var):
    '''
    Make cross tab of type of crime and year.
    Input:
        df: Pandas DF
        var: str
    Output:
        Pandas DF
    '''
    mean_by_year1 = df1.groupby('year').agg({var: 'mean'})
    mean_by_year2 = df2.groupby('year').agg({var: 'mean'})
    mean_by_year3 = df3.groupby('year').agg({var: 'mean'})
    mean_by_year = pd.merge(mean_by_year1, mean_by_year2, left_index=True,
                            right_index=True)
    mean_by_year = pd.merge(mean_by_year, mean_by_year3, left_index=True,
                            right_index=True)
    mean_by_year.columns = ['Chicago', 'NYC', 'Charleston']

    # cross_var_year.rename(columns={'All' : 'Total'}, index={'All' : 'Total'},\
    #     inplace = True)
    # cross_var_year['Perc Change'] = (cross_var_year[2018] / 
    #                                   cross_var_year[2017] - 1) * 100
    # cross_var_year['Perc Change'] = cross_var_year['Perc Change'].round(2)
    # cross_var_year = cross_var_year[['Total', 2017, 2018, 'Perc Change']]
    # cross_var_year.replace(float('inf'), np.NaN, inplace = True)
    # cross_var_year.sort_values('Total', ascending = False)
    # cross_var_year.index = cross_var_year.index.str.capitalize()
    # cross_var_year.rename_axis(var.upper().replace("_", " "), inplace = True)
    # cross_var_year.rename_axis("", axis = 1,  inplace = True)
    #cross_class_year.sort_values('Total', ascending = False, inplace = True)

    return mean_by_year

def describe(eviction_df, var_type=None, variable=None, by_year=False):
    '''
    Get descriptive stats of the variables that belong to var_type.
    Inputs:
        eviction_df: Pandas DataFrame
        var_type: 'demographics', 'real-estate' or 'evictions' (string)
        variable: Specific variable
    '''
    var_classification = {
        'demographics': ['population', 'poverty-rate', 'median-household-income',
                         'pct-white', 'pct-af-am', 'pct-hispanic', 'pct-am-ind',
                         'pct-asian', 'pct-nh-pi', 'pct-multiple', 'pct-other'],
        'real-estate': ['renter-occupied-households', 'pct-renter-occupied',
                        'median-gross-rent', 'median-property-value',
                        'rent-burden'],
        'evictions': ['eviction-filings', 'evictions', 'eviction-rate',
                      'eviction-filing-rate']}
    if variable:
        if by_year:
            return eviction_df.groupby(year, variable).agg('mean')

    if not var_type:
        print(eviction_df.describe())
    else:
        print(eviction_df[var_classification[var_type]].describe())


def make_bar_plot(describe_df):
    describe_df.plot.bar()
    plt.title('Eviction rate')
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


def see_scatterplot(eviction_df, xcol, ycol, colorcol=None, logx=False,
                    logy=False, xjitter=False, yjitter=False):
    '''
    Print scatterplot of columns specified of the eviction df. If color column
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
    plt.title('Scatterplot of eviction DataFrame \n {} and {}'
              .format(xcol, ycol))
    plt.show()


def see_histograms(eviction_df, geo_area, col, years=None, restrict=None):
    '''
    Produce histograms of the numeric columns in credit_df. If columns is
    specified, it produces histograms of those columns. If restrict dictionary
    is specified, restricts to the values inside the percentile range specified.
    Inputs:
        credit_df: Pandas DataFrame
        col: str
        yeard: [int]
    Output:
        Individual Graphs (Num of graphs = Num of numeric cols)
    '''
    plt.clf()
    figs = {}
    axs = {}
    if not years:
        years = range(2000, 2017, 4)
    for year in years:
        col_to_plot = eviction_df.loc[eviction_df['year'] == year, col]
        if col_to_plot.isna().all():
            continue
        if restrict:
            min_val = col_to_plot.quantile(restrict[0])
            max_val = col_to_plot.quantile(restrict[1])
            col_to_plot = col_to_plot.loc[(col_to_plot <= max_val)
                             & (col_to_plot >= min_val)]

        num_bins = min(20, col_to_plot.nunique())

        figs[col] = plt.figure()
        axs[col] = figs[col].add_subplot(111)
        n, bins, patches = axs[col].hist(col_to_plot, num_bins,
                                         facecolor='blue', alpha=0.5)
        title = geo_area + ', '+ str(year) + "\n" + " ".join(col.split("-"))
        axs[col].set_title(title)
    plt.show()
