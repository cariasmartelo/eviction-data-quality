##Machine Learning for Public Policy.
##Angelica Valdiviezo
##Chi Nguyen
##Camilo Arias
##Code to download data from eviction lab

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import geopandas as gpd
from shapely.geometry import Point, Polygon, MultiPoint, shape


def download_from_api(x):
	'''
	Download data from API
	'''
	pass

def import_from_csv(csv_file):
	'''
	Import eviction data from csv file.
	Inputs:
		csv_file: String
	Output:
		Pandas DataFrame
	'''
	eviction_df = pd.read_csv(csv_file)
	return eviction_df