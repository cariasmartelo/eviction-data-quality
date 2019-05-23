
'''
CAPP 30254 1 Machine Learning for Public Policy
Utility functions for HW1
Spring 2019
Professor: Rayid Ghani
Camilo Arias
'''
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sodapy import Socrata
import geopandas as gpd
from census import Census
from us import states
from shapely.geometry import Point, Polygon, MultiPoint, shape


CHICAGO_OPEN_DATA = "data.cityofchicago.org"
CRIME_ENDPOINT = "6zsd-86xi"
COMMUNITY_ENDPOINT = "igwz-8jzy"
TRACT_ENDPOINT = "74p9-q2aq"
CENSUS_KEY = '54e41f7f22600865e7900fed1a621533282df1ae'

#Crime classification from:
#http://gis.chicagopolice.org/clearmap_crime_sums/crime_types.html
CRIME_CLASS = {
    'Violent Crime': ['01A', '02', '03', '04A', '04B'],
    'Property Crime': ['05', '06', '07', '09'],
    'Less serious offences': ['01B', '08A', '08B', '10', '11', '12', '13',\
                              '14', '15', '16', '17', '18', '19', '20', '22',\
                              '24', '26']
}

ACS_TABLES_KEYS = {'B01001_001E': 'total_population',
                  'B01001H_001E': 'white_population',
                  'B07011_001E': 'median_income',
                  'B06003_010E': 'foreign_born',
                  'B14006_002E': 'below_poverty'}


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
    acs_df.loc[acs_df['median_income'] < 0, 'median_income'] = np.NaN
    acs_df.loc[acs_df['total_population'] == 0, 'total_population'] = np.NaN
    acs_df['total_population'] = pd.to_numeric(acs_df['total_population'],
                                               errors = "coerce")
    for col in acs_tables_dict.values():
        acs_df[col + '_ratio'] = acs_df[col] / acs_df['total_population']

    return acs_df


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


def load_crime_data(limit, year_from, year_to):
    '''
    Load 2017 and 2018 Chicago crime data from City of Chicago Open Data portal
    using Socrata API from year_from to year_to
    Input:
        Limit: int
        year_from: int
        year_to: int
    Output:
        Pandas Data Frame
    '''

    client = Socrata(CHICAGO_OPEN_DATA, None)
    crime = client.get(CRIME_ENDPOINT, where=("year >= {} and year <= {}"
                                              .format(year_from, year_to)),
                       limit=limit)

    crime_df = pd.DataFrame.from_dict(crime)

    #Seting up types
    for col in ['year', 'ward', 'y_coordinate', 'x_coordinate', 'latitude',\
                'longitude']:
        crime_df[col] = pd.to_numeric(crime_df[col], errors = "coerce")

    for col in ['arrest', 'domestic']:
        crime_df[col] = crime_df[col].astype(bool)

    for col in ['date', 'updated_on']:
        crime_df[col] = pd.to_datetime(crime_df[col])

    #Adding clasification column
    crime_class_inv = {}
    for k, v in CRIME_CLASS.items():
        for code in v:
            crime_class_inv[code] = k

    crime_df['crime_class'] = crime_df['fbi_code'].map(crime_class_inv)

    return crime_df


def make_cross_var_year(df, var):
    '''
    Make cross tab of type of crime and year.
    Input:
        df: Pandas DF
        var: str
    Output:
        Pandas DF
    '''
    cross_var_year = pd.crosstab(df[var], df.year, margins = True)
    cross_var_year.rename(columns={'All' : 'Total'}, index={'All' : 'Total'},\
        inplace = True)
    cross_var_year['Perc Change'] = (cross_var_year[2018] / 
                                      cross_var_year[2017] - 1) * 100
    cross_var_year['Perc Change'] = cross_var_year['Perc Change'].round(2)
    cross_var_year = cross_var_year[['Total', 2017, 2018, 'Perc Change']]
    cross_var_year.replace(float('inf'), np.NaN, inplace = True)
    cross_var_year.sort_values('Total', ascending = False)
    cross_var_year.index = cross_var_year.index.str.capitalize()
    cross_var_year.rename_axis(var.upper().replace("_", " "), inplace = True)
    cross_var_year.rename_axis("", axis = 1,  inplace = True)
    #cross_class_year.sort_values('Total', ascending = False, inplace = True)

    return cross_var_year


def graph_crimes_year(df, crime_class):
    '''
    Bar graph of type of crime of classificaiton given by year
    Input:
        df: Pandas DF
        crime_class: str
    Output:
        Displays graph
    Addapted code from
    https://matplotlib.org/gallery/lines_bars_and_markers/barchart.html
    https://preinventedwheel.com/easy-matplotlib-bar-chart/
    '''
    cross_tab = pd.crosstab([df.crime_class, df.primary_type], df.year)
    df = cross_tab.loc[crime_class] 
    ind = np.arange(len(df))  # the x locations for the groups
    width = 0.35  # the width of the bars
    fig, ax = plt.subplots()
    rects1 = ax.bar(ind - width/2, df[2017], width,
                    color='SkyBlue', label=2017)
    rects2 = ax.bar(ind + width/2, df[2018], width,
                    color='IndianRed', label=2018)

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Freq')
    ax.set_title("{}s in Chicago (2017 "
                 "and 2018) and % change"
                 .format(crime_class.replace("_", " ")))
    ax.set_xticks(ind)
    ax.set_xticklabels(df.index.str.capitalize())
    ax.legend()
    plt.xticks(rotation=90)
    #plt.tight_layout()

    perc_change = ((df[2018] / df[2017] - 1) * 100).round(2)
    perc_change = ["{} %".format(x) for x in perc_change]
    pairs = len(df)
    make_pairs = zip(*[ax.get_children()[:pairs],ax.get_children()\
                 [pairs:pairs*2]])
    for i,(left, right) in enumerate(make_pairs):
        ax.text(i,max(left.get_bbox().y1,right.get_bbox().y1)+2,
                perc_change[i], horizontalalignment ='center')

    ax.get_yaxis().set_major_formatter(plt.FuncFormatter(lambda x,
                                       loc: "{:,}".format(int(x))))
    ax.patch.set_facecolor('#FFFFFF')
    ax.spines['bottom'].set_color('#CCCCCC')
    ax.spines['bottom'].set_linewidth(1)
    ax.spines['left'].set_color('#CCCCCC')
    ax.spines['left'].set_linewidth(1)

    plt.show()

def load_community_area_data():
    '''
    Load 2017 and 2018 Chicago crime data from City of Chicago Open Data portal
    using Socrata API.
    Input:
        Limit: int
    Output:
        Pandas Data Frame
    '''

    client = Socrata(CHICAGO_OPEN_DATA, None)
    community_area= client.get(COMMUNITY_ENDPOINT)

    community_area_df = pd.DataFrame.from_dict(community_area)
    community_area_df.rename(columns={'the_geom' : 'location',
                                      'area_num_1' : 'community_area'}, 
                                      inplace = True)

    return community_area_df


def convert_to_geopandas(df):
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


    df['geometry'] = df['geometry'].apply(shape_)
    geo_df = gpd.GeoDataFrame(df, crs = 'epsg:4326', geometry = df['geometry'])

    return geo_df


def map_crimes_loc(geo_comm, geo_crime, filer_col, filter_vals, color_col = None):
    '''
    Produce a map of the crimes location in Chicago over a neighborhood map.
    Inputs:
        geo_comm: GeoPandasDf
        geo_crime: GeoPandasDf
        filter_col: Str
        filter_vals: [str]
        color_col: str
    '''
    plt.clf()
    base = geo_comm.plot(color='white', edgecolor='black')
    if not color_col:
        geo_crime[geo_crime[filer_col] == filter_vals].plot(ax=base, marker='.',
            color = 'red', markersize=4, legend = True)
    else:
        geo_crime[geo_crime[filer_col] == filter_vals].plot(ax=base, marker='.',
            column = color_col, markersize=4, legend = True)

    plt.title('{}s in Chicago (2017 and 2018)'.format(filter_vals.capitalize()))
    plt.show()
    plt.clf()


def merge_tract_census(tract_df, census_df):
    '''
    Merge tract dataframe with census tract dataframe.
    Inputs:
        tract_df = Pandas DF
        census_df = Pandas Df
    Output:
        Pandas DF
    '''

    tract_merged = tract_df.merge(census_df, on = 'tract')
    return tract_merged

def join_tract_census_crime(geo_trac, geo_crime):
    '''
    Spatial Join between geo_trac and geo_crime
    Inputs:
        geo_trac: Geopandas
        geo_crime: Geopandas
    Output:
        Geopandas
    '''
    geo_crime = geo_crime[geo_crime.geometry.notna()] 
    tract_census_crime = gpd.sjoin(geo_trac, geo_crime, how="inner",
                                         op='intersects')
    return tract_census_crime



def merge_comm_crime(com_df, crime_df, groupby_col):
    '''
    Merge community dataset with crime dataset, grouping by the groupby_col.
    Inputs:
        com_df: Pandas Df
        crime_df: Pandas DF
        groupby_col: str
    Output:
        Pandas DF
    '''

    com_area_crime = com_df.merge(crime_df.groupby(['community_area',
                                                   groupby_col],
                                    as_index = False).size().unstack(),
                                    on = 'community_area')
    return com_area_crime


def make_cross_comm_crime_year(comm_df, crime_df, crime = None):
    '''
    Make cross tab by community area of selected type of crime
    Inputs:
        comm_df: DataFrame
        crime_df: DataFrame
        crime: str
    Output:
        DataFrame
    '''
    if not crime:
        cross_crime_year = crime_df.groupby(['community_area', 'year'],
                           as_index = False).size().unstack()
    else:
        cross_crime_year = crime_df[crime_df['crime_class'] == crime]\
                       .groupby(['community_area', 'year'],
                       as_index = False).size().unstack()

    cross_crime_year['Total'] = cross_crime_year[2017] + cross_crime_year[2018]
    cross_crime_year['Perc Change'] = (cross_crime_year[2018] /
                                       cross_crime_year[2017] - 1) * 100
    merged = comm_df.merge(cross_crime_year, on = 'community_area')

    return merged



def map_comm_crime(geo_com_area_crime, crime_type):
    '''
    Produce a map of the community areas in Chicago by number of crimes.
    Inputs:
        geo_com_area_crime: GeoPandasDf
        crime_type: str
    ''' 

    geo_com_area_crime.plot(column = crime_type, cmap = 'Reds', legend = True)
    plt.title('{}s in Chicago (2017 and 2018)'.format(crime_type.capitalize()))
    plt.show()
    plt.clf()

