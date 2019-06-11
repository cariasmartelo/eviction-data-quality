import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mapclassify
import geopandas as gpd

FIGURES_FOLDER = "figures/"

def plot_map(eviction_gdf, variable, year, save_fig=False):
    '''
    Map by zip code the value of the column indicated in colorcol and the year.
    Inputs:
        eviction_gdf: GeoDataFrame
        variable: Str
        year: int or list
    Output:
        Map
    '''
    fig, ax = plt.subplots(figsize=(8, 12))
    eviction_gdf.plot(color="grey", ax=ax, edgecolor="black")
    eviction_gdf.loc[(eviction_gdf['year'] == year) &\
                     (eviction_gdf[variable].notna())].plot(ax=ax,
                                                            column=variable,
                                                            cmap='viridis_r',
                                                            scheme='quantiles',
                                                            legend=True)
    ax.set_title('Tracts of Chicago by {} in {}\n(Tracts without data'
                     ' in grey)'.format(" ".join(variable.split("_")),
                                        year))
    plt.axis('off')
    if save_fig:
        fname = "{}map_{}_{}.png".format(FIGURES_FOLDER, variable, year)
        plt.savefig(fname)

    plt.show()

def plot_top_10pct_tracts(gdf, variable, years, save_fig=False):
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
        gdf_plt = gdf[(gdf.year == year) & gdf[variable].notna()]

        fig, ax = plt.subplots(figsize=(6, 10))
        ax.set_aspect('equal')
        gdf_plt.plot(ax=ax, color='white', edgecolor='grey')
        gdf_plt[gdf_plt[variable].rank(pct=True) > .90].plot(ax=ax, color="m")
        ax.set_title('Tracts of Chicago that are in the top 10 percent\n of {} in {}'\
                     .format(" ".join(variable.split("_")), year))
        plt.axis('off')
        if save_fig:
            fname = "{}map_top10_{}_{}.png".format(FIGURES_FOLDER, variable, year)
            plt.savefig(fname)
        plt.show();

def plot_top_10pct_tracts_ktimes(gdf, variable, k, get_tracts=False, save_fig=False):
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
    ax.set_title('Tracts of Chicago that are in the top 10 percent of\n{} at least {}times in the period 2010-2017'\
                 .format(" ".join(variable.split("_")), k))
    plt.axis('off')
    if save_fig:
        fname = "{}map_top10_{}times_{}.png".format(FIGURES_FOLDER, k, variable)
        plt.savefig(fname)
    plt.show();

    if get_tracts:
        return tracts
