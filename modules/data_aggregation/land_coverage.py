
import geopandas as gpd
import warnings #This is just because the gpd.append in the split_GeoDataFrame_smaller_pixels comes up with a warning
import matplotlib.pyplot as plt
import pandas as pd
from shapely.geometry import Polygon
from modules.ml_logic.utils import get_corners, slice_picture_coords
import fiona
from modules.data_aggregation.building_data import get_full_landuse_coverage
from ast import literal_eval
import os
import json


def get_data(city):
    '''
    Input: city name capitalised
    Output: GeoPandas DataFrame read from the .geojson
    '''

    input_path = os.path.join("data", "processed_data", city, f"gpd_{city.lower()}.geojson")
    gpd_df = gpd.read_file(input_path)

    gpd_df['geometry_coordinates'] = [list(gpd_df.geometry[index].exterior.coords)[::-1] for index, row in gpd_df.iterrows()]
    return gpd_df


def get_data_clipped(gpd_df: gpd, polygon = Polygon([(2.264216,48.813898),(2.264216,48.900502),(2.42172,48.900502),(2.42172,48.813898),(2.264216,48.813898)])):
    '''
    Inputs:
    gpd_df: geopandas dataframe to be clipped
    polygon: shapely.geometry Polygon class, which will clip the geopandas Dataframe. The current default is for Paris bbox
    Output:
    Clipped geopandas DataFrame
    '''
    polygon =  polygon
    return gpd.clip(gpd_df, polygon)

def split_GeoDataFrame_smaller_pixels(gpd_df, pixels, chunk_size, testing = True):
    '''
    Inputs:
    - gpd_df is a geopandas dataframe,
    - pixels is a list of bounding boxes (small clips), multiple can and should be passed
    - testing option True or False, if true, the function will only run for pixels[1000: 1000+ chunk_size]
    Output:
    gpd_df, the polygons of the original dataframe have been clipped to the pixels in order to assess land cover use by pixel
    '''
    import json
    start_idx = 1000
    gpd_smaller_pixels = gpd.GeoDataFrame(columns = list(gpd_df.columns))
    step = 0
    if testing == True:
        while start_idx < 1150:
            for clip_pixel in pixels[start_idx:(start_idx + chunk_size)]:
                new = gpd.clip(gpd_df, Polygon(clip_pixel))
                with warnings.catch_warnings():
                    warnings.simplefilter(action='ignore', category=FutureWarning)
                    gpd_smaller_pixels = gpd_smaller_pixels.append(gpd.GeoDataFrame(new))
            start_idx += chunk_size
    elif testing == False:
        for clip_pixel in pixels:
                # print(start_idx + chunk_size)
            new = gpd.clip(gpd_df, Polygon(clip_pixel))
            with warnings.catch_warnings():
                warnings.simplefilter(action='ignore', category=FutureWarning)
                gpd_smaller_pixels = gpd_smaller_pixels.append(gpd.GeoDataFrame(new))
    if len(gpd_smaller_pixels)> len(gpd_df):
        print('Success !')
    return gpd_smaller_pixels


INPUT_PATH = os.path.join('data','processed_data')
def get_landuse(gpd_df, city):
    '''
    Inputs:
    - city name has to be capitalised
    - gpd_df is the gpd on which the landuse is contained
    This method will import the original dataframe onto which this landuse information will be appended
    The output is this original dataframe with the new columns
    '''
    import_path = os.path.join(INPUT_PATH, city ,f'{city}.csv')
    working_df = pd.read_csv(import_path)
    df = working_df.copy()
    df['bb'] = working_df.bb.apply(literal_eval)
    new_columns = ['zone_1', 'zone_2', 'zone_3', 'zone_4', 'zone_5', 'zone_6', 'zone_7', 'zone_8', 'zone_9', 'zone_10', 'zone_11']
    for index, column in enumerate(new_columns):
        zone = gpd_df[gpd_df['c_zonage'] == index+1].geometry
        df[column] = df['bb'].apply(lambda row: get_full_landuse_coverage(landuse_polygons=zone, pixel= row))
    df['sum'] = df.loc[:, new_columns].sum(axis = 1)
    return df

def get_map(gpd_df):
    '''
    Returns the mapped data of the geopandas dataframe
    '''
    map = gpd_df.plot(column = 'categories', figsize = (20,20), legend = True, cmap = 'viridis')
    return map


def split_Geo_df_and_get_landuse(index, gpd_df, pixels, city):
    '''
    Provide the index of the chunk of the pixels df you would
    like to get the landuse details, this function uses the split_GeoDataFrame_smaller_pixels and will save
    the output of that function as a csv in data/processed_data/city/processed_landuse
    '''
    for idx in index:
        start_index = idx*1000
        print('getting the data for slices: ', start_index, start_index +1000)
        part_1 = split_GeoDataFrame_smaller_pixels(gpd_df=gpd_df, pixels = pixels[start_index:start_index+1000], chunk_size = 100, testing =False)
        print('getting the landuse for slices: ', start_index, start_index+1000)
        part_2 = get_landuse(part_1, city)
        print('saving the csv for slices: ', start_index, start_index+1000)
        part_2.to_csv(f'data/processed_data/{city}/processed_landuse/{city}_landuse_{start_index}_{start_index+1000}.csv')



def clean_dfs_crop(df, start_index):
    '''
    Will drop the necessary columns from the dataframe and will select, df[start_index: start_index +1000] rows
    '''
    if 'Unnamed: 0' in df.columns:
        print('dropping column ...')
        df.drop(columns = 'Unnamed: 0', inplace=True)
    else: print('no need to drop!')
    df = df.iloc[start_index: start_index+1000]
    return df
