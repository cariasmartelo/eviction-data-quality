##Machine Learning for Public Policy.
##Angelica Valdiviezo
##Chi Nguyen
##Camilo Arias
##Code to download data from eviction lab.
import boto3
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import geopandas as gpd
from shapely.geometry import Point, Polygon, MultiPoint, shape


def download_from_api(state, geo_level='all', filepath=None, download_dict=False):
    '''
    Download data using Amazon S3 API
    Inputs:
        U.S. state: State 2 letter code (str)
        geo_level: 'all', cities', 'counties', 'states' or 'tracts' (str)
        filepath: filepath: Path to store data (str). Default input folder.
        download_dict: True to download data dictionary.
    '''
    if not filepath:
        filepath = (r'/Users/camiloariasm/Google Drive/Escuela/MSCAPP/Q3/ML/'
                    r'eviction-data-quality/inputs/eviction/')
    if not os.path.exists(filepath):
        os.mkdir(filepath)
    if not os.path.exists(filepath + state + r'/'):
        os.mkdir(filepath + state + r'/')
    files = [geo_level + '.csv']
    if not geo_level == 'all':
        files.append(geo_level + '.geojson')
    s3 = boto3.client('s3')
    for file in files:
        s3.download_file('eviction-lab-data-downloads', state + r'/' + file,
                         filepath + state + r'/'+ file)
        print("Downloaded {} of {} in {}".format(file, state, filepath + state))
    if download_dict:
        s3.download_file('eviction-lab-data-downloads', 'DATA_DICTIONARY.txt',
                         filepath + 'DATA_DICTIONARY.txt')
        print("Downloaded {} in {}".format('DATA_DICTIONARY.txt', filepath))
    

def import_csv(csv_file):
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

    eviction_df = pd.read_csv(csv_file, dtype = eviction_lab_dtypes)

    return eviction_df


def import_geojson(gpd_file):
    '''
    Import eviction data from geojson file.
    Inputs:
        gpd_file: String
    Output:
        GPD DataFrame
    '''
    eviction_gdf = gpd.read_file(gpd_file)

    return eviction_gdf

def describe(eviction_df, var_type=None):
    '''
    Get descriptive stats of the variables that belong to var_type.
    Inputs:
        eviction_df: Pandas DataFrame
        var_type: 'demographics', 'real-estate' or 'evictions' (string)
    '''
    var_classification = {
        'demographics': ['population', 'poverty-rate','median-household-income',
                         'pct-white', 'pct-af-am', 'pct-hispanic', 'pct-am-ind',
                         'pct-asian', 'pct-nh-pi', 'pct-multiple', 'pct-other'],
        'real-estate': ['renter-occupied-households', 'pct-renter-occupied',
                        'median-gross-rent', 'median-property-value',
                        'rent-burden'],
        'evictions': ['eviction-filings', 'evictions', 'eviction-rate',
                      'eviction-filing-rate']}
    if not var_type:
        print(eviction_df.describe())
    else:
        print(eviction_df[var_classification[var_type]].describe())

def map(eviction_gdf, variable, year):
    '''
    Map by zip code the value of the column indicated in colorcol and the year.
    Inputs:
        eviction_gdf: GeoDataFrame
        variable: Str
        year: int
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

    ax = eviction_gdf.plot(color="grey")
    eviction_gdf.dropna().plot(ax=ax, column=colorcol, cmap='viridis',
                             legend=True)

    ax.set_title('Tracts of Illinois by {} in {}\n(Tracts without data'
                 ' in grey)'.format(" ".join(variable.split("-")), year))
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


