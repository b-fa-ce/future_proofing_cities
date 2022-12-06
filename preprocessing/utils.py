import h5py
import glob
import numpy as np
import pandas as pd
import geopandas
from shapely.geometry import Point, Polygon

import os

# input and output file paths
INPUT_PATH = os.path.join('..','raw_data')
OUTPUT_PATH = os.path.join('..','processed_data')

# lon, lat city bounding boxes
CITY_BOUNDING_BOXES = {'Paris': [[2.264216,2.42172],[48.813898,48.900502]],
                  'Berlin': [[13.313794,13.471299],[52.475607,52.55571]],
                  'London': [[-0.216408,-0.058904],[51.469149,51.551072]],
                  'Brussels': [[4.288857,4.446361],[50.808751,50.891855]]
}


def import_ECOSTRESS_data(file_path: str):
    """
    imports hd5-file data from ECOSTRESS
    """
    return h5py.File(file_path, 'r')


def import_city_data(city_name: str)-> tuple:
    """
    returns np.arrays of lst, lat, lon, height data
    for a given city where city_name folder in
    raw_data/city_name
    """

    folder_path = os.path.join(INPUT_PATH, city_name,'')
    lst_path = glob.glob(folder_path + 'ECOSTRESS_L2*')[0]
    geo_path = glob.glob(folder_path + 'ECOSTRESS_L1*')[0]

    lst = np.array(import_ECOSTRESS_data(lst_path)['SDS']['LST'])
    lat = np.array(import_ECOSTRESS_data(geo_path)['Geolocation']['latitude'])
    lon = np.array(import_ECOSTRESS_data(geo_path)['Geolocation']['longitude'])
    height = np.array(import_ECOSTRESS_data(geo_path)['Geolocation']['height'])

    return lst, lat, lon, height


def correct_value_percentage(lst_data: np.array) -> float:
    """
    returns pertentage of correct values, i.e. non-zero values
    """
    return np.count_nonzero(lst_data)/np.multiply(*lst_data.shape)


def convert_to_Celsius(lst_data):
    """
    converts LST data to Celcius and replaces
    missing values with -1000
    """

    # access LST, multiply by 0.02 scale factor
    # see https://ecostress.jpl.nasa.gov/downloads/psd/ECOSTRESS_SDS_PSD_L2_ver1-1.pdf
    # convert to celcius
    SCALE_FACTOR = 0.02
    KELVIN_CELCIUS = -273.15

    # missing values, i.e. zero values
    lst_data[lst_data == 0] = (-1000 - KELVIN_CELCIUS) * 1/SCALE_FACTOR

    return (SCALE_FACTOR * lst_data) + KELVIN_CELCIUS


def convert2df_coord(lst_data: np.array, lat_data: np.array, lon_data: np.array,\
                    height_data: np.array, lon_lim: list, lat_lim: list) -> pd.DataFrame:
    """
    converts input data: LST, Latitude, Longitude, height in hd5 format
    with lon_lim, lat_lim: longitude & latitude boundaries
    to Pandas DataFrame
    """

    # convert LST to Celcius
    lst = convert_to_Celsius(lst_data)

    # flatten
    lst_flat = lst.reshape(1,np.multiply(*lst.shape))
    lat_flat = lat_data.reshape(1,np.multiply(*lat_data.shape))
    lon_flat = lon_data.reshape(1,np.multiply(*lon_data.shape))
    height_flat = height_data.reshape(1,np.multiply(*height_data.shape))

    # difference to average LST
    lst_av = np.average(lst_flat[lst_flat != -1000])
    lst_diff = lst_flat - lst_av

    # differene to average height
    height_av = np.average(height_flat[lst_flat != -1000])
    height_diff = height_flat - height_av


    # combine arrays
    lst_geo = np.vstack((lst_flat, lat_flat, lon_flat,\
                         height_flat, lst_diff, height_diff)).T

    # remove missing values -> leave for now
    # lst_geo = lst_geo[lst_geo[:,0]!=-1000]

    # latitude
    lst_geo = lst_geo[(lst_geo[:,1]>= lat_lim[0]) & (lst_geo[:,1]<= lat_lim[1])]

    # longitude
    lst_geo = lst_geo[(lst_geo[:,2]>= lon_lim[0]) & (lst_geo[:,2]<= lon_lim[1])]



    # convert to Pandas DataFrame

    df = geopandas.GeoDataFrame(lst_geo)
    df.columns = ['LST','Latitude', 'Longitude', 'height', 'LST_Difference', 'height_Difference']

    return df



def save_as_csv(city_name: str):
    """
    saves DataFrames to csv files
    """

    df = convert2df_coord(*import_city_data(city_name),\
        *CITY_BOUNDING_BOXES[city_name])

    out_path = os.path.join(OUTPUT_PATH,f'{city_name}.csv')
    df.to_csv(out_path, index=False)


if __name__ == '__main__':
    for city in CITY_BOUNDING_BOXES:
        save_as_csv(city)
