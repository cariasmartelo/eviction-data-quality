URL = 'https://eviction.lcbh.org/sites/default/files/attachments/Chicago-Evictions-data-Release-1-2019May16.zip'
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

import pandas as pd
import requests
import zipfile
import io

def download_evict_lawyers_data(filename=EVICT_FILENAME, filepath=None):
    '''
    Load Chicago Eviction data from the Chicago Evictions project
    by the Lawyers' Committee for Better Housing
    Input:
        - filename (str): name of the file that contains the Eviction
            data for tracts.

    Output:
        df
    '''
    # TODO this is the part we have to go back if we solve the automating the zip thing
    # r = requests.get(URL) # i get a 404 error 

    #here a have to be able to call the ZIP folder, this can manage the rest
    # with ZipFile(zip_filename, 'r') as zip: 
    #     data = zip.read('eviction_data_tract.csv')
    evict_df = pd.read_csv(filename, use_cols=TO_USE, dtype=D_TYPES, parse_dates=PARSE_DATES)

    return evict_df