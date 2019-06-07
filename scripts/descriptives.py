import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mapclassify
import geopandas as gpd



def plot_map(eviction_gdf, variable, years, change = False):
    '''
    Map by zip code the value of the column indicated in colorcol and the year.
    Inputs:
        eviction_gdf: GeoDataFrame
        variable: Str
        year: int or list
    Output:
        Map
    '''
    if isinstance(years, int):
        year = years
        fig, ax = plt.subplots(figsize=(8, 12))
        eviction_gdf.plot(color="grey", ax=ax, edgecolor="black")
        eviction_gdf.loc[(eviction_gdf['year'] == year) &\
                        (eviction_gdf[variable].notna())].plot(ax=ax, 
                                                               column=variable,
                                                               cmap='viridis',
                                                               scheme='quantiles',
                                                               legend=True)

        ax.set_title('Tracts of Chicago by {} in {}\n(Tracts without data'
                     ' in grey)'.format(" ".join(variable.split("_")),
                                        year))
    else:
        fig, ax = plt.subplots(nrows = len(years), figsize=(8, 12))
        for i, year in enumerate(years):
            eviction_gdf.plot(color="grey", ax=ax[i], edgecolor="black")
            eviction_gdf.loc[(eviction_gdf['year'] == year) &\
                            (eviction_gdf[variable].notna())].plot(ax=ax[i], 
                                                                   column=variable,
                                                                   cmap='viridis',
                                                                   scheme='quantiles',
                                                                   legend=True)

            ax[i].set_title('Tracts of Chicago by {} in {}\n(Tracts without data'
                         ' in grey)'.format(" ".join(variable.split("_")),
                                            year))
    plt.show()
