
import geopandas as gpd
import warnings #This is just because the gpd.append in the split_GeoDataFrame_smaller_pixels comes up with a warning
import matplotlib.pyplot as plt
import pandas as pd
from shapely.geometry import Polygon
from modules.ml_logic.utils import get_corners, slice_picture_coords


def get_data():
    '''
    From the preprocessed_data directory read the .geojson file as a geopandas dataframe
    '''
    gpd_paris = gpd.read_file("../../data/processed_data/Paris/gpd_paris.geojson")
    gpd_paris['geometry_coordinates'] = [list(gpd_paris.geometry[index].exterior.coords)[::-1] for index, row in gpd_paris.iterrows()]
    return gpd_paris


def get_data_clipped(gpd_df, polygon = Polygon([(2.264216,48.813898),(2.264216,48.900502),(2.42172,48.900502),(2.42172,48.813898),(2.264216,48.813898)])):
    '''
    The current default is for Paris bbox
    '''
    polygon =  polygon
    return gpd.clip(gpd_df, polygon)

def split_GeoDataFrame_smaller_pixels(gpd_df, slice_coords):
    '''
    Inputs:
    - gpd_df is a geopandas dataframe,
    - full_coords are the bounding coordinates of the geopandas_df,
    - scaling_factor is the number of new output xs and ys
    Output:
    gpd_df with the new amount of rows
    '''
    clipped_pixels = get_corners(slice_coords = slice_coords)
    gpd_smaller_pixels = gpd.GeoDataFrame(columns = list(gpd_df.columns))
    step = 0
    for clip in clipped_pixels:
        new = gpd.clip(gpd_df, Polygon(clip))
        with warnings.catch_warnings():
            warnings.simplefilter(action='ignore', category=FutureWarning)
            gpd_smaller_pixels = gpd_smaller_pixels.append(gpd.GeoDataFrame(new))
        step += 1
        if step%100 == 0:
            print(f"step = {step}")
    if len(gpd_smaller_pixels)> len(gpd_df):
        print('Success !')
    return gpd_smaller_pixels




def get_map(gpd_df):
    '''
    Returns the mapped data of the geopandas dataframe
    '''
    map = gpd_df.plot(column = 'categories', figsize = (20,20), legend = True, cmap = 'viridis')
    return map


def split_data(gpd_df):
    '''
    Returns X dataframes where x is the amount of different landuse categories
    '''
    x_1 = gpd_df[gpd_df['c_zonage'] == 1]
    x_2 = gpd_df[gpd_df['c_zonage'] == 2]
    x_3 = gpd_df[gpd_df['c_zonage'] == 3]
    x_4 = gpd_df[gpd_df['c_zonage'] == 4]
    x_5 = gpd_df[gpd_df['c_zonage'] == 5]
    x_6 = gpd_df[gpd_df['c_zonage'] == 6]
    x_7 = gpd_df[gpd_df['c_zonage'] == 7]
    x_8 = gpd_df[gpd_df['c_zonage'] == 8]
    x_9 = gpd_df[gpd_df['c_zonage'] == 9]
    x_10 = gpd_df[gpd_df['c_zonage'] == 10]
    x_11 = gpd_df[gpd_df['c_zonage'] == 11]

    return x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9, x_10, x_11
