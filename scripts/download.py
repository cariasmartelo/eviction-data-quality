'''
Machine Learning for Public Policy.
Angelica Valdiviezo
Chi Nguyen
Camilo Arias
Code to download data from eviction lab.
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
                2015: {
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
                2009: {
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


######### CHI OP DATA ############

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


def download_crime_data(year_from=2008, year_to=2016, limit=3000000, filepath=None):
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

    crime_df = download_chiopdat_data(API_ENDPOINTS['CRIME_ENDPOINT'], year_from, year_to,
                                  limit=limit)

    #Seting up types

    to_numeric = ['year', 'inspection_number', 'street_number', 'property_group',\
                  'latitude', 'longitude']
    to_integer = ['arrest', 'domestic']

    to_datetime = ['violation_date', 'violation_status_date',\
                   'violation_last_modified_date']
    crime_df = do_transformations(crime_df, to_numeric, to_datetime, to_integer)

    #Adding clasification column
    crime_class_inv = {}
    for k, v in CRIME_CLASS.items():
        for code in v:
            crime_class_inv[code] = k
    crime_df['crime_class'] = crime_df['fbi_code'].map(crime_class_inv)

    crime_df.to_csv(os.path.join(filepath, 'crime.csv'))
    crime_df.sample(frac=0.05).to_csv(os.path.join(filepath, 'sample_crime.csv'))
    print("Downloaded crime of Chicago in {}"
          .format(filepath))


def download_building_violation_data(year_from=2008, year_to=2016, limit=3000000,
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
    for year in [2009, 2015]:
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
    cols_to_keep = [level for level in acs_tables_dict[2015] if level != 'total']
    cols_to_keep += ['year', 'tract', 'county', 'pop_over_25']
    joint_df[cols_to_keep].to_csv(os.path.join(filepath, 'education.csv'))
    print("Downloaded Education Attainment of Chicago in {}"
          .format(filepath))


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

######## MERGE THE DATABASES #########
def load_building_violations():
    '''
    load and clean
    '''
    pass

def load_crime_data():
    '''
    load and clean
    '''
    pass

def load_education():
    '''
    load and clean
    '''
    pass

def load_acs_data():
    '''
    load and clean
    '''
    pass

def join_bases():
    '''
    '''
    EVICT_FILENAME = #TODO
    D_TYPES = {'tract': str}
    PARSE_DATES = ['filing_year']
    TO_USE = ['filing_year', 'tract', 'eviction_filings_total',
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

    evict_df = pd.read_csv(filename, use_cols=TO_USE, dtype=D_TYPES, parse_dates=PARSE_DATES)

    # join with acs, education crime, building violations



if __name__ == "__main__":
    # download_eviction_data()
    download_crime_data()
    download_building_violation_data()
    download_census_data()


