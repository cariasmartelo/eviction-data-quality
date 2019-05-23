'''
Machine Learning for Public Policy.
Angelica Valdiviezo
Chi Nguyen
Camilo Arias
Code to download data from eviction lab.
'''
import os
import boto3
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mapclassify
import geopandas as gpd
from sodapy import Socrata
from census import Census
from us import states
from shapely.geometry import Point, Polygon, MultiPoint, shape


CENSUS_KEY = '54e41f7f22600865e7900fed1a621533282df1ae'

CHICAGO_OPEN_DATA = "data.cityofchicago.org"
API_ENDPOINTS = {
    "CRIME_ENDPOINT": "6zsd-86xi",
    "COMMUNITY_ENDPOINT": "igwz-8jzy",
    "TRACT_ENDPOINT": "74p9-q2aq",
    "ZIP_CODE_BOUNDARIES": 'unjd-c2ca'
}

CRIME_CLASS = {
    'Violent Crime': ['01A', '02', '03', '04A', '04B'],
    'Property Crime': ['05', '06', '07', '09'],
    'Less serious offences': ['01B', '08A', '08B', '10', '11', '12', '13',\
                              '14', '15', '16', '17', '18', '19', '20', '22',\
                              '24', '26']}

ACS_TABLES_KEYS = {'B01001_001E': 'total_population',
                  'B01001H_001E': 'white_population',
                  'B07011_001E': 'median_income',
                  'B06003_010E': 'foreign_born',
                  'B14006_002E': 'below_poverty'}

def download_eviction_data(state, geo_level='all', filepath=None, download_dict=False):
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
    if geo_level != 'all':
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


def load_chiopdat_data(api_endpoint, year_from=None, year_to=None, limit=10000):
    '''
    Load data from Chicago Open Data portal using Socrata API and the api_endpoint. If
    limit is specified, load no more than limit number of observations.
    Input:
        api_endpoint: str
        year_from: int
        year_to: int
        limit: int
    Output:
        Pandas Data Frame
    '''

    client = Socrata(CHICAGO_OPEN_DATA, None)
    if not year_from:
        data_dict = client.get(api_endpoint, limit=limit)
    else:
        data_dict = client.get(api_endpoint, where=("year >= {} and year <= {}"
                               .format(year_from, year_to)), limit=limit)

    data_df = pd.DataFrame.from_dict(data_dict)
    if 'the_geom' in data_df.columns:
        data_df.rename(columns={'the_geom' : 'location'}, 
                                inplace = True)
    return data_df


def load_census_data(acs_tables_dict):
    '''
    Download census data at the tract level of the table keys indicated.
    Inputs:
        keys: (str)
    Output

    '''
    keys = tuple([k for k,v in acs_tables_dict.items()])
    
    c = Census(CENSUS_KEY)
    acs_download = c.acs5.get(keys,
        {'for': 'tract:*', 'in': 'state:{} county:{}'.format('17', '031')})

    acs_df = pd.DataFrame(acs_download)
    acs_df.rename(columns = acs_tables_dict, inplace = True)
    # acs_df.loc[acs_df['median_income'] < 0, 'median_income'] = np.NaN
    # acs_df.loc[acs_df['total_population'] == 0, 'total_population'] = np.NaN
    # acs_df['total_population'] = pd.to_numeric(acs_df['total_population'],
    #                                            errors = "coerce")
    # for col in acs_tables_dict.values():
    #     acs_df[col + '_ratio'] = acs_df[col] / acs_df['total_population']

    return acs_df


def load_crime_data(year_from, year_to, limit=500000):
    '''
    Load 2017 and 2018 Chicago crime data from City of Chicago Open Data portal
    using Socrata API from year_from to year_to
    Input:
        year_from: int
        year_to: int
        limit: int

    Output:
        Pandas Data Frame
    '''

    crime_df = load_chiopdat_data(API_ENDPOINTS['CRIME_ENDPOINT'], year_from, year_to,
                                  limit)

    #Seting up types
    for col in ['year', 'ward', 'y_coordinate', 'x_coordinate', 'latitude',\
                'longitude']:
        crime_df[col] = pd.to_numeric(crime_df[col], errors = "coerce")

    for col in ['arrest', 'domestic']:
        crime_df[col] = crime_df[col].astype(int)

    for col in ['date', 'updated_on']:
        crime_df[col] = pd.to_datetime(crime_df[col])

    #Adding clasification column
    crime_class_inv = {}
    for k, v in CRIME_CLASS.items():
        for code in v:
            crime_class_inv[code] = k

    crime_df['crime_class'] = crime_df['fbi_code'].map(crime_class_inv)

    return crime_df



def load_tract_shapefile():
    '''
    Download shapefile data and create GeoPandasDataFrame.
    Output:
        DataFrame
    '''
    client = Socrata(CHICAGO_OPEN_DATA, None)
    tract_area = client.get(TRACT_ENDPOINT)

    tract_area_df = pd.DataFrame.from_dict(tract_area)
    tract_area_df.rename(columns={'the_geom' : 'location',
                                      'tractce10' : 'tract'},
                                      inplace = True)
    return tract_area_df


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

