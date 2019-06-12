'''
Code to Download data for eviction proyect.
Group 2
'''
import os
import boto3
import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mapclassify
import geopandas as gpd
from sodapy import Socrata
from census import Census
from us import states
from ast import literal_eval
from shapely.geometry import Point, Polygon, MultiPoint, shape


CENSUS_KEY = '54e41f7f22600865e7900fed1a621533282df1ae'

CHICAGO_OPEN_DATA = "data.cityofchicago.org"
API_ENDPOINTS = {
    "CRIME_ENDPOINT": "6zsd-86xi",
    "BUILDING_VIOLATIONS": "22u3-xenr",
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

ACS_TABLES_KEYS = {
                2017: {
                    'total':
                    {'B15003_001E': 'pop_over_25'},
                    'up_to_middle':
                        {'B15003_002E': 'no_schooling_completed',
                         'B15003_003E': 'nursery_school_completed',
                         'B15003_004E': 'kiderdarten_completed',
                         'B15003_005E': 'first_grade',
                         'B15003_006E': 'second_grade',
                         'B15003_007E': 'third_grade',
                         'B15003_008E': 'fourth_grade',
                         'B15003_009E': 'fifth_grade',
                         'B15003_010E': 'sixth_grade',
                         'B15003_011E': 'seventh_grade',
                         'B15003_012E': 'eigth_grade'},
                    'not_highschool_grad':
                        {'B15003_013E': 'ninth_grade',
                         'B15003_014E': 'tenth_grade',
                         'B15003_015E': 'eleventh_grade',
                         'B15003_016E': 'twelfth_grade'},
                    'high_school_grad':
                        {'B15003_017E': 'highschool_completed',
                         'B15003_018E': 'GED_credential'},
                    'some_college':
                        {'B15003_019E': 'college_less_one_year',
                         'B15003_020E': 'college_more_one_year'},
                    'associate':
                        {'B15003_021E': 'associate_degree'},
                    'bachelor':
                        {'B15003_022E': 'bachelors_degree'},
                    'graduate':
                        {'B15003_023E': 'masters_degree',
                         'B15003_024E': 'professional_School_degree',
                         'B15003_025E': 'doctorate_degree'}},
                2010: {
                    'total':
                        {'B15002_001E': 'pop_over_25'},
                    'total_by_gdr':
                        {'B15002_002E': 'male_over_25',
                         'B15002_019E': 'female_over_25'},  
                    'up_to_middle':
                        {'B15002_003E': 'm_no_schooling_completed',
                         'B15002_004E': 'm_nursery_school_completed',
                         'B15002_005E': 'm_5_and_6_grade',
                         'B15002_006E': 'm_7_and_8_grade',
                         'B15002_020E': 'f_no_schooling_completed',
                         'B15002_021E': 'f_nursery_school_completed',
                         'B15002_022E': 'f_5_and_6_grade',
                         'B15002_023E': 'f_7_and_8_grade'},
                    'not_highschool_grad':
                        {'B15002_007E': 'm_ninth_grade',
                         'B15002_008E': 'm_tenth_grade',
                         'B15002_009E': 'm_eleventh_grade',
                         'B15002_010E': 'm_twelfth_grade',
                         'B15002_024E': 'f_ninth_grade',
                         'B15002_025E': 'f_tenth_grade',
                         'B15002_026E': 'f_eleventh_grade',
                         'B15002_027E': 'f_twelfth_grade'},
                    'high_school_grad':
                        {'B15002_011E': 'm_highschool_GED_completed',
                         'B15002_028E': 'f_highschool_GED_completed'},
                    'some_college':
                        {'B15002_012E': 'm_college_less_one_year',
                         'B15002_013E': 'm_college_more_one_year',
                         'B15002_029E': 'f_college_less_one_year',
                         'B15002_030E': 'f_college_more_one_year'},
                    'associate':
                        {'B15002_014E': 'm_associate_degree',
                         'B15002_031E': 'f_associate_degree'},
                    'bachelor':
                        {'B15002_015E': 'm_bachelors_degree',
                         'B15002_032E': 'f_bachelors_degree'},
                    'graduate':
                        {'B15002_016E': 'm_masters_degree',
                         'B15002_017E': 'm_professional_school_degree',
                         'B15002_018E': 'm_doctorate_degree',
                         'B15002_033E': 'f_masters_degree',
                         'B15002_034E': 'f_professional_school_degree',
                         'B15002_035E': 'f_doctorate_degree'}
                    }
                }

######### EVICTION LAB ############

# TODO maybe delete
# def download_eviction_data(state='IL', geo_level='all', filepath=None, download_dict=False):
#     '''
#     Download data using Amazon S3 API
#     Inputs:
#         U.S. state: State 2 letter code (str)
#         geo_level: 'all', cities', 'counties', 'states' or 'tracts' (str)
#         filepath: filepath: Path to store data (str). Default input folder.
#         download_dict: True to download data dictionary.
#     '''
#     if not filepath:
#         filepath = os.path.join(os.getcwd(), 'eviction', '')
#     if not os.path.exists(filepath):
#         os.mkdir(filepath)
#     files = [geo_level + '.csv']
#     if geo_level != 'all':
#         files.append(geo_level + '.geojson')
#     s3 = boto3.client('s3')
#     for file in files:
#         s3.download_file('eviction-lab-data-downloads', os.path.join(state, file),
#                          os.path.join(filepath, file))
#         print("Downloaded {} of {} in {}"
#               .format(file, state, filepath))
#     if download_dict:
#         s3.download_file('eviction-lab-data-downloads', 'DATA_DICTIONARY.txt',
#                          os.path.join(filepath, 'DATA_DICTIONARY.txt'))
#         print("Downloaded {} in {}".format('DATA_DICTIONARY.txt', filepath))


######### DOWNLOADS ############

def download_chiopdat_data(api_endpoint, year_from=None, year_to=None,
                           date_column='year', timestamp=False, limit=10000):
    '''
    Load data from Chicago Open Data portal using Socrata API and the api_endpoint. If
    limit is specified, load no more than limit number of observations. To limit the 
    dates, it needs the date_column and whether it is a timestamp column or an integer.
    Default is integer.
    Input:
        api_endpoint: str
        year_from: int
        year_to: int
        date_column: int
        timestamp: bool
        limit: int
    Output:
        Pandas Data Frame
    '''

    client = Socrata(CHICAGO_OPEN_DATA, None)
    if not year_from:
        data_dict = client.get(api_endpoint, limit=limit)
    else:
        if timestamp:
            data_dict = client.get(api_endpoint,
                                   where=("date_extract_y({}) BETWEEN {} and {}"
                                   .format(date_column, year_from, year_to)),
                                   limit=limit)
        else:
            data_dict = client.get(api_endpoint,
                                   where=("{} BETWEEN {} and {}"
                                   .format(date_column, year_from, year_to)),
                                   limit=limit)

    data_df = pd.DataFrame.from_dict(data_dict)
    if 'the_geom' in data_df.columns:
        data_df.rename(columns={'the_geom' : 'location'}, 
                                inplace = True)

    return data_df

def do_transformations(df, to_numeric, to_datetime, to_integer):
    '''
    Transform variables in DF to type.
    Inputs:
        to_numeric, to_datetime, to_integer = []
    Output:
        DF
    '''
    for col in to_numeric:
        df[col] = pd.to_numeric(df[col], errors = "coerce")

    for col in to_integer:
        df[col] = df[col].astype(int)

    for col in to_datetime:
        df[col] = pd.to_datetime(df[col])

    return df


def download_crime_data(year_from=2010, year_to=2017, limit=3000000, filepath=None):
    '''
    Load Chicago crime data from City of Chicago Open Data portal
    using Socrata API from year_from to year_to
    Input:
        year_from: int
        year_to: int
        limit: int

    Output:
        saves data in csv file
    '''
    if not filepath:
        filepath = os.path.join(os.getcwd(), 'ch_opdat', '')
    if not os.path.exists(filepath):
        os.mkdir(filepath)

    crime_df = download_chiopdat_data(API_ENDPOINTS['CRIME_ENDPOINT'],
                                      year_from, year_to, limit=limit)

    #Seting up types

    to_numeric = ['id', 'year', 'latitude', 'longitude']
    to_integer = ['arrest', 'domestic']
    to_datetime = ['date']
    crime_df = do_transformations(crime_df, to_numeric,
                                  to_datetime, to_integer)

    #Adding clasification column
    crime_class_inv = {}
    for k, v in CRIME_CLASS.items():
        for code in v:
            crime_class_inv[code] = k
    crime_df['crime_class'] = crime_df['fbi_code'].map(crime_class_inv)

    crime_df.to_csv(os.path.join(filepath, 'crime.csv'))
    crime_df.sample(frac=0.05).to_csv(os.path.join(filepath,
                                                   'sample_crime.csv'))
    print("Downloaded crime of Chicago in {}".format(filepath))


def download_building_violation_data(year_from=2010, year_to=2017, limit=3000000,
                                     filepath=None):
    '''
    Load Building Violation data from City of Chicago Open Data portal
    using Socrata API from year_from to year_to
    Input:
        year_from: int
        year_to: int
        limit: int

    Output:
        saves data in csv file
    '''
    if not filepath:
        filepath = os.path.join(os.getcwd(), 'ch_opdat', '')
    if not os.path.exists(filepath):
        os.mkdir(filepath)

    building_df = download_chiopdat_data(API_ENDPOINTS['BUILDING_VIOLATIONS'],
                                         year_from, year_to,
                                         date_column='violation_date',
                                         timestamp=True, limit=limit)

    to_numeric = ['inspection_number', 'street_number', 'property_group',\
                  'latitude', 'longitude']
    to_integer = []

    to_datetime = ['violation_date', 'violation_status_date',\
                   'violation_last_modified_date']

    building_df = do_transformations(building_df, to_numeric, to_datetime,
                                     to_integer)
    building_df['year'] = building_df['violation_date'].map(lambda x: x.year)

    building_df.to_csv(os.path.join(filepath, 'building_viol.csv'))
    building_df.sample(frac=0.05).to_csv(os.path.join(filepath,
                                         'sample_building_viol.csv'))

    print("Downloaded Building Violations of Chicago in {}"
          .format(filepath))


def download_census_data(acs_tables_dict=ACS_TABLES_KEYS, filepath=None):
    '''
    Download census data at the tract level of the table keys indicated.
    Inputs:
        keys: (str)
    Output

    '''
    if not filepath:
        filepath = os.path.join(os.getcwd(), 'acs', '')
    if not os.path.exists(filepath):
        os.mkdir(filepath)

    joint_df = pd.DataFrame()    
    for year in [2010, 2017]:
        keys = []
        for level, key_pairs in acs_tables_dict[year].items():
            keys += [k for k in key_pairs]
        keys = tuple(keys)
        c = Census(CENSUS_KEY, year=year)
        acs_download = c.acs5.get(keys,
            {'for': 'tract:*', 'in': 'state:{} county:{}'.format('17', '031')})

        acs_df = pd.DataFrame(acs_download)

        for level, key_pairs in acs_tables_dict[year].items():
            acs_df.rename(columns = key_pairs, inplace = True)
            if level == 'total':
                continue
            sublevels = [column for k, column in key_pairs.items()]
            acs_df[sublevels] = acs_df[sublevels].div(acs_df['pop_over_25'], axis=0)
            acs_df[level] = acs_df[sublevels].sum(axis=1)
        acs_df['year'] = year
        joint_df = pd.concat([joint_df, acs_df])
    cols_to_keep = [level for level in acs_tables_dict[2017] if level != 'total']
    cols_to_keep += ['year', 'tract', 'county', 'pop_over_25']
    joint_df[cols_to_keep].to_csv(os.path.join(filepath, 'education.csv'))
    print("Downloaded Education Attainment of Chicago in {}"
          .format(filepath))


def download_tract_shapefile(filepath=None):
    '''
    Download shapefile data and create GeoPandasDataFrame.
    Output:
        DataFrame
    '''
    if not filepath:
        filepath = os.path.join(os.getcwd(), 'ch_opdat', '')
    if not os.path.exists(filepath):
        os.mkdir(filepath)
    
    client = Socrata(CHICAGO_OPEN_DATA, None)
    tract_area = client.get(API_ENDPOINTS['TRACT_ENDPOINT'])
    tract_area_df = pd.DataFrame.from_dict(tract_area)
    tract_area_df.rename(columns={'the_geom' : 'location',
                                  'tractce10' : 'tract'},
                                  inplace = True)
    tract_area_df.to_csv(os.path.join(filepath, 'tracts.csv'))
    print("Downloaded shape of Chicago tracts in {}"
          .format(filepath))




######## Prepare each DF for the merge


def load_tract_shapefile(csv_file):
    '''
    load and clean
    Used to make the spatial joins.
    '''
    tracts_df = pd.read_csv(csv_file)
    tracts_df = tracts_df[['location', 'tract', 'commarea']]
    tracts_df['location'] = tracts_df['location'].map(lambda x :
        literal_eval(x) if not isinstance(x, float) else x)
    return tracts_df


def load_building_violations_spread(csv_file):
    '''
    load and clean
    It has the building violations by lat and long.
    '''
    build_viol = pd.read_csv(csv_file)

    to_numeric = ['inspection_number', 'street_number', 'property_group',\
                  'latitude', 'longitude']
    to_integer = []

    to_datetime = ['violation_date', 'violation_status_date',\
                   'violation_last_modified_date']
    build_viol = do_transformations(build_viol, to_numeric, to_datetime, to_integer)
    build_viol['location'] = build_viol['location'].map(lambda x :
        literal_eval(x) if not isinstance(x, float) else x)

    return build_viol

def aggregate_building_data(bv_df, tracts_df, save=False, filepath=None):
    '''
    Produce the aggregation of Crime Data by tract and year
    Inputs:
        crime_df = Pandas DF of crime
        tracts_df= Pandas DF of tracts
    Output:
        Pandas DF
        Saves to inputs.
    '''
    geo_bv = convert_to_geopandas(bv_df, 'location')
    geo_tract = convert_to_geopandas(tracts_df, 'location')
    tract_bv = join_with_tract(geo_tract, geo_bv)
    bv_agg = aggregate(tract_bv, ['violation_status', 'inspection_category',\
                                  'department_bureau'])
    bv_agg.rename(columns={'total': 'total_building_violations'}, inplace = True)

    if save:
        bv_agg.to_csv(os.path.join(filepath, 'building_violation_by_tract.csv'))

    return bv_agg


def load_crime_data_spread(csv_file):
    '''
    load and clean
    '''
    crime_df = pd.read_csv(csv_file)
    to_numeric = ['id', 'year', 'latitude', 'longitude']
    to_integer = ['arrest', 'domestic']
    to_datetime = ['date', 'updated_on']
    crime_df = do_transformations(crime_df, to_numeric, to_datetime, to_integer)
    crime_df['location'] = crime_df['location'].map(lambda x :
        literal_eval(x) if not isinstance(x, float) else x)

    return crime_df

def aggregate_crime_data(crime_df, tracts_df, save=False, filepath=None):
    '''
    Produce the aggregation of Crime Data by tract and year
    Inputs:
        crime_df = Pandas DF of crime
        tracts_df= Pandas DF of tracts
    Output:
        Pandas DF
        Saves to inputs.
    '''
    geo_crime = convert_to_geopandas(crime_df, 'location')
    geo_tract = convert_to_geopandas(tracts_df, 'location')
    tract_crime = join_with_tract(geo_tract, geo_crime)
    crime_agg = aggregate(tract_crime, ['crime_class', 'primary_type',\
                                        'domestic', 'arrest'])
    crime_agg.rename(columns={'total': 'total_crime'}, inplace = True)

    if save:
        crime_agg.to_csv(os.path.join(filepath, 'crime_by_tract.csv'))

    return crime_agg


def aggregate(tract_df, columns_to_aggregate):
    '''
    Make aggregations of df by tract and year. Columns to aggregate
    are the columns to produce counts by. If 'crime_class' in columns
    to aggregate, the resulting df will will have total by crime class.
    Inputs:
        tract_df: Joined df with tract and data
        columns_to_aggregate: [str]
    Output:
        Pandas DF
    '''
    df = tract_df.groupby(['tract', 'year']).size().reset_index()
    df.rename(columns = {0:'total'}, inplace = True)
    for col in columns_to_aggregate:
        agg_by_col_df = (tract_df.groupby(['tract', 'year', col])
                               .size().unstack().add_prefix('total_' + col + '_')
                               .fillna(0).reset_index())
        df = df.merge(agg_by_col_df, on=['tract', 'year'])    

    return df


def convert_to_geopandas(df, location_col):
    '''
    Converts the pandas dataframe to geopandas DataFrame
    Inputs:
        df: Pandas DataFrame
        location_col = stri
    Output:
        Geopandas DataFrame
    '''
    def shape_(x):

        '''
        Convert JSON location attribute to shapely.
        '''
        if not x:
            return x
        if isinstance(x, float):
            return np.NaN
        if isinstance(x, dict):
            if 'type' in x:
                return shape(x)
            else:
                return Point(float(x['longitude']),float(x['latitude']))


    df['geometry'] = df[location_col].map(shape_)
    geo_df = gpd.GeoDataFrame(df, crs = 'epsg:4326',
                              geometry = df['geometry'])

    return geo_df


def join_with_tract(geo_tract, geo_df):
    '''
    Spatial Join between geo_trac and geo_crime
    Inputs:
        geo_trac: Geopandas
        geo_crime: Geopandas
    Output:
        Geopandas
    '''
    geo_tract = geo_tract[['location', 'tract']]
    geo_df = geo_df[geo_df.geometry.notna()] 
    geo_tract_df = gpd.sjoin(geo_tract, geo_df, how="inner",
                                         op='intersects')

    return geo_tract_df


def load_acs_data(acs_filename):
    '''
    load and clean
    Load of ACS data that only has 2010 and 2017.
    '''
    # TODO ANGELICA
    acs_df = pd.read_csv(acs_filename)
    return acs_df


def impute_acs_data(df, acs=True, save=False, filepath=None, year_dict=None):
    '''
    impute acs data so we get one row per year
    Used to create the education_year_tract and acs_year_tract files
    '''

    new_df = pd.DataFrame(columns=df.columns)
    empty_row = [""] * len(df.columns)
    if not year_dict:
        if acs:
            years_dict = {"2006-2010 5-year estimates": [2010, 2011, 2012, 2013,\
                                                         2014, 2015, 2016, 2017],
                          "2013-2017 5-year estimates": [2018]}
        else:
            years_dict = {2010: [2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017],
                          2017: [2018]}

    i = 0
    rows, _ = df.shape
    for row in range(rows):
        years = years_dict[df.iloc[row]["year"]]
        for year in years:
           # here it is doubling the columns to the dataframe, don’t understand why
           new_df = new_df.append(pd.Series(empty_row), ignore_index=True)
           new_df.iloc[i] = df.iloc[row]
           new_df.iloc[i]["year"] = year
           i += 1

    # temporary fix to the columns that are being created
    to_drop = [i for i in range(len(df.columns))]
    new_df = new_df.drop(columns=to_drop)
   
    if save:
        if acs:
            new_df.to_csv(os.path.join(filepath, 'acs_year_tract.csv'))
        else:
            new_df.to_csv(os.path.join(filepath, 'educ_year_year.csv'))


    return new_df


########################## Load prepared data for Merge and do some modifs.

def load_crime(csv_crime_csv):
    '''
    '''
    crime_df = pd.read_csv(csv_crime_csv)
    crime_df['tract'] = crime_df['tract'].astype(str)
    crime_df['tract'] = crime_df['tract'].apply(lambda x: '{0:0>6}'.format(x))
    subcrimes = crime_df.columns[4:7]  
    crimes_percentage = crime_df[subcrimes].div(crime_df['total_crime'], axis=0)
    crimes_percentage = crimes_percentage.add_prefix('perc_')
    crime_df = pd.concat([crime_df, crimes_percentage], axis=1)
    to_increase = crime_df.columns[3:7]
    crimes_increase = crime_df.sort_values(['tract', 'year'])\
                        .set_index(['tract', 'year'])\
                        [to_increase].pct_change()
    crimes_increase = crimes_increase.applymap(lambda x: 1 if x >= 1 else x)
    crimes_increase = crimes_increase.add_prefix('perc_increase_')
    crimes_increase = crimes_increase.reset_index()
    crimes_increase.loc[crimes_increase.index[crimes_increase['year'] == 2010],2:]\
        = np.NaN
    crime_df = crime_df.merge(crimes_increase, on = ['tract', 'year'])
    crime_df.drop('Unnamed: 0', axis = 1, inplace=True)

    return crime_df


def load_building(csv_building_merged):
    '''
    '''
    building_viol = pd.read_csv(csv_building_merged)
    building_viol['tract'] = building_viol['tract'].astype(str)
    building_viol['tract'] = building_viol['tract'].apply(lambda x: '{0:0>6}'.format(x)) 
    subviolations = building_viol.columns[4:8]
    building_viol.rename(columns={'total_bioldinv_violations':'total_building_violations'}, inplace=True)
    bv_percentage = building_viol[subviolations].div(building_viol['total_building_violations'], axis=0)
    bv_percentage = bv_percentage.add_suffix('_perc')
    building_viol = pd.concat([building_viol, bv_percentage], axis=1)
    to_increase = 'total_building_violations'
    building_increase = building_viol.sort_values(['tract', 'year'])\
                         .set_index(['tract', 'year'])\
                         [to_increase].pct_change()
    building_increase = building_increase.rename('perc_increase_bv')
    building_increase = building_increase.reset_index()
    building_increase.loc[building_increase['year'] == 2010, building_increase.columns[2:]] = np.NaN
    building_viol = building_viol.merge(building_increase, on = ['tract', 'year'])
    building_viol.drop('Unnamed: 0', axis = 1, inplace=True)
    return building_viol


def load_acs(acs_filename):
    '''
    load and clean
    '''
    # TODO ANGELICA
    acs_df = pd.read_csv(acs_filename)
    acs_df['tract'] = acs_df['tract'].astype(str)
    acs_df['tract'] = acs_df['tract'].map(lambda x: x[-6:])
    housing_type = ['housing_units_rental', 'housing_units_other']
    acs_df[housing_type]= acs_df[housing_type].apply\
                         (lambda x: x/acs_df['housing_units_total'])
    population_types = (['population_poverty_below', 'population_poverty_above',
                         'population_race_white', 'population_race_black',
                         'population_race_latinx', 'population_race_asian',
                         'population_race_other'])
    acs_df[population_types]= acs_df[population_types].apply\
                              (lambda x: x/acs_df['population_total'])
    def get_mayority(df):
        max_concentration = df[['population_race_black', 'population_race_white',\
                                'population_race_latinx', 'population_race_asian']].max()
        if max_concentration < 0.4:
            return 'Integrated'
        else:
            if df['population_race_black'] == max_concentration:
                return 'Black'
            elif df['population_race_white'] == max_concentration:
                return 'White'
            elif df['population_race_latinx'] == max_concentration:
                return 'Latin'
            elif df['population_race_asian'] == max_concentration:
                return 'Asian'
    acs_df['race'] = acs_df.apply(get_mayority, axis=1)

    def get_poor(df):
        if df['population_poverty_above'] > 0.3:
            return 'Poor'
        else:
          return 'Non-Poor'

    acs_df['poor'] = acs_df.apply(get_poor, axis=1)
    acs_df = acs_df.drop('Unnamed: 0', axis=1)
    return acs_df

def load_education(education_filename):
    '''
    load education
    '''
    education_df = pd.read_csv(education_filename)
    education_df['tract'] = education_df['tract'].astype(int)
    education_df['tract'] = education_df['tract'].apply(lambda x: '{0:0>6}'.format(x)) 
    education_df['tract'] = education_df['tract'].astype(str)
    education_df = education_df.drop(['Unnamed: 0', 'Unnamed: 0.1'], axis=1)
    return education_df

def load_evict(evict_csv):
    evict_filename = evict_csv
    d_type = {'tract': str}
    parse_date = ['filing_year']
    to_use = ['filing_year', 'tract', 'eviction_filings_total',
           'eviction_filings_rate', 'eviction_filings_completed',
           'case_type_single_action', 'case_type_joint_action', 'back_rent_median',
           'back_rent_0', 'back_rent_1_to_999', 'back_rent_1000_to_2499',
           'back_rent_2500_to_4999', 'back_rent_5000_or_more',
           'landlord_represented', 'tenant_represented',
           'tenant_rep_pa', 'tenant_rep_laa', 'eviction_order_yes',
           'eviction_order_no', 'eviction_order_yes_tenant_prose',
           'eviction_order_no_tenant_prose',
           'eviction_order_yes_tenant_represented',
           'eviction_order_no_tenant_represented',
           'eviction_order_yes_tenant_rep_pa', 'eviction_order_no_tenant_rep_pa',
           'eviction_order_yes_tenant_rep_laa', 'eviction_order_no_tenant_rep_laa',
           'ftu_eviction_order', 'ftu_other_outcome', 'ftu_no_outcome',
           'ftu_eviction_order_tenant_prose', 'ftu_other_outcome_tenant_prose',
           'ftu_no_outcome_tenant_prose', 'ftu_eviction_order_tenant_represented',
           'ftu_other_outcome_tenant_represented',
           'ftu_no_outcome_tenant_represented', 'default_eviction_order_yes',
           'default_eviction_order_no', 'default_eviction_order_yes_tenant_prose',
           'default_eviction_order_no_tenant_prose',
           'default_eviction_order_yes_tenant_represented',
           'default_eviction_order_no_tenant_represented']

    evict_df = pd.read_csv(evict_filename, usecols=to_use, dtype=d_type, parse_dates=parse_date)
    evict_df['year'] = evict_df['filing_year'].map(lambda x: x.year)
    evict_df['tract'] = evict_df['tract'].map(lambda x: x[-6:])

    
    return evict_df

def load_tract(tract_shp, keep_geom=False):
    '''
    load tract
    '''
    cols_to_keep = ['tract', 'commarea']
    if keep_geom:
        cols_to_keep += ['location']
    tract = load_tract_shapefile(tract_shp)
    tract = tract[cols_to_keep]
    tract['tract'] = tract['tract'].apply(lambda x: '{0:0>6}'.format(x))
    return tract

#####################Join dfs
def join_bases(eviction_df, acs_df, education_df, crime_df, building_viol_df,
               tract_df, gpds=False):
    '''
    Join dfs
    '''
    return_df = pd.merge(eviction_df, acs_df, on = ['tract', 'year'])
    return_df = pd.merge(return_df, education_df, on = ['tract', 'year'])
    return_df = pd.merge(return_df, crime_df, on = ['tract', 'year'])
    return_df = pd.merge(return_df, building_viol_df, on = ['tract', 'year'])
    if not gpds:
        return_df = pd.merge(return_df, tract_df, on = 'tract')
    else:
        return_df = pd.merge(tract_df, return_df, on = 'tract')
    mean_by_commarea = return_df.groupby(['commarea', 'year']).mean().reset_index()  
    return_df = pd.merge(return_df, mean_by_commarea, on = ['commarea', 'year'],
                         suffixes = ('', '_mean_by_commarea'))
    return return_df
    


if __name__ == "__main__":
    download_eviction_data()
    download_crime_data()
    download_building_violation_data()
    download_census_data()
    tract_data = load_tract('ch_opdat/tracts.csv')
    crime_data = load_crime_data_spread('ch_opdat/crime.csv')
    bv_data = load_building_violations_spread('ch_opdat/building_viol.csv')
    aggregate_crime_data(crime_data, tract_data, save=True)
    aggregate_building_data(bv_df, tract_data, save=True)
    acs_data = load_acs_data('acs/census_data_tract.csv')
    impute_acs_data(acs_data, acs=True, save=True, filepath=None, year_dict=None)
    educ_data = load_acs_data('acs/censu_data_tract.csv')
    impute_acs_data(educ_data, acs=False, save=True, filepath=None, year_dict=None)

    pass