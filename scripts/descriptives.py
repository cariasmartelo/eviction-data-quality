import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mapclassify
import geopandas as gpd



def plot_map(eviction_gdf, variable, years):
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
    plt.axis('off')
    plt.show()


def plot_map_change(eviction_gdf, variable, years):
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
    plt.axis('off')
    plt.show()

def plot_top_10pct_tracts(gdf, variable, years):
    '''
    Plots the tracts in Chicago that were in the top 10% of a given variable,
    for a given list of years.
    Inputs:
        - gdf (GeoDataFrame)
        - years (lst)
        - variable (str)
    '''
    if isinstance(years, str):
        years = list(years)

    for year in years:
        gdf_plot = gdf[(gdf.year == year) & gdf[variable].notna()]

        fig, ax = plt.subplots(figsize=(6, 10))
        ax.set_aspect('equal')
        gdf_plot.plot(ax=ax, color='white', edgecolor='grey')
        gdf_plot[gdf_plot[variable].rank(pct=True) > .90].plot(ax=ax)
        ax.set_title('Tracts of Chicago that are in the top 10% percent of {} in {}'\
                     .format(" ".join(variable.split("_")), year))
        plt.axis('off')
        plt.show();

def plot_top_10pct_tracts_ktimes(gdf, variable, k, get_tracts=False):
    '''
    '''
    temp_df = gdf.assign(perc=gdf.groupby("year")[variable].rank(pct=True))
    temp_df.loc[temp_df.perc >= .90, 'top_10'] = 1
    temp_df.loc[temp_df.perc < .90, 'top_10'] = 0
    counts = temp_df[['top_10', 'tract']].groupby('tract').sum()
    top_k_tracts = counts[counts.top_10 > k]
    tracts = list(top_k_tracts.index)


    fig, ax = plt.subplots(figsize=(6, 10))
    ax.set_aspect('equal')
    gdf.plot(ax=ax, color='white', edgecolor='grey')
    gdf[(gdf.tract.isin(tracts)) & gdf[variable].notna()].plot(ax=ax)
    ax.set_title('Tracts of Chicago that are in the top 10% percent of {} at least {} times in the period 2010-2017'\
                 .format(" ".join(variable.split("_")), k))
    plt.axis('off')
    plt.show();

    if get_tracts:
        return tracts



