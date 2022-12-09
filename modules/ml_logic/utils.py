import numpy as np
import pandas as pd
from ast import literal_eval
import os

from modules.data_aggregation.params import CITY_BOUNDING_BOXES

# processed data
INPUT_PATH = os.path.join('..','..','data','processed_data')


def slice_picture_coords(full_coords: list, scaling_factor: int, overlap_percent: int = 0) -> list:
    """
    description:
    returns a list of length scaling_factor^2
    of coordinates of the slices in format: [[lon,lon+lon_step], [lat,lat+lat_step]]

    input:
    bounding box full_coords = [[lon_1, lon_2], [lat_1, lat_2]],
    scaling_factor: number of divisions in lon and lat
    overlap_percent: percentage of overlap between individual tiles

    output:
    list of [[lon,lon+lon_step], [lat,lat+lat_step]]

    usage: slice_picture_coords(CITY_BOUNDING_BOXES['Paris'], 100)
    """

    # flatten bb cooordinates
    lon1, lon2, lat1, lat2 = [item for sublist in full_coords for item in sublist]

    # lat and lon distance
    lat_dist = abs(lat1 - lat2)
    lon_dist = abs(lon1 - lon2)

    # step width
    lat_step = lat_dist/scaling_factor
    lon_step = lon_dist/scaling_factor

    # overlap in lon, lat
    lat_overlap = overlap_percent/100 * lat_step
    lon_overlap = overlap_percent/100 * lon_step

    # create coordinates of tiles
    tiles_coords = [[[lon,lon+lon_step], [lat,lat+lat_step]] for lat in np.arange(lat1,lat2, lat_step-lat_overlap) \
        for lon in np.arange(lon1,lon2, lon_step-lon_overlap)]

    return tiles_coords


def get_all_sub_coords(full_df: pd.DataFrame,
                       num_px_lon: int = 10,
                       num_px_lat: int = 6,
                       overlap_percent: int = 0) -> list:
    """
    description:
    returns the sub image according to the smaller
    tile coordinates

    input:
    full_df: full original dataframe
    tiles_coords: coordinates of all subtiles from slice_picture_coords()
    image_number: number pixels in longitude and latitude

    output:
    returns the image_number-th subtile dataframe
    """

    # drop irrelavant features
    full_df.drop(columns=['LST', 'ele','ll_corner', 'ur_corner', 'bb'])


    # convert imported upper left corner str into list
    ul_import = full_df.ul_corner.apply(literal_eval)
    lr_import = full_df.lr_corner.apply(literal_eval)

    # divide ul_corner into lists of lat and lon
    ul_lon = np.array([ul[0] for ul in ul_import])[:,0]
    ul_lat = np.array([ul[0] for ul in ul_import])[:,1]

    # divide lr_corner into lists of lat and lon
    lr_lon = np.array([lr[0] for lr in lr_import])[:,0]
    lr_lat = np.array([lr[0] for lr in lr_import])[:,1]

    total_num_tiles = int(len(full_df)/(num_px_lon* num_px_lat))

    tiles_coords = []

    for i in range(total_num_tiles):

        # get pixel size
        lat_size, lon_size = lr_lat[i] - ul_lat[i], lr_lon[i] - ul_lon[i]

        # step size
        step_lat = num_px_lat * lat_size
        step_lon = num_px_lon * lon_size


        # divide slice_coords into lists of lat and lon
        slice_bound_lat = [ul_lat[i], ul_lat[i] + step_lat]
        slice_bound_lon = [ul_lon[i], ul_lon[i] + step_lon]

        tiles_coords.append([slice_bound_lon,slice_bound_lat])


    return tiles_coords


def get_sub_tile(full_df: pd.DataFrame, tiles_coords: list, image_number: int) -> pd.DataFrame:
    """
    description:
    returns the sub image according to the smaller
    tile coordinates

    input:
    full_df: full original dataframe
    tiles_coords: coordinates of all subtiles from slice_picture_coords()
    image_number: number of image to return

    output:
    returns the image_number-th subtile dataframe
    """
    # convert imported upper left corner str into list
    ul_import = full_df.ul_corner.apply(literal_eval)
    lr_import = full_df.lr_corner.apply(literal_eval)

    # divide ul_corner into lists of lat and lon
    ul_lon = np.array([ul[0] for ul in ul_import])[:,0]
    ul_lat = np.array([ul[0] for ul in ul_import])[:,1]

    # divide lr_corner into lists of lat and lon
    lr_lon = np.array([lr[0] for lr in lr_import])[:,0]
    lr_lat = np.array([lr[0] for lr in lr_import])[:,1]

    # divide slice_coords into lists of lat and lon
    slice_bound_lon = tiles_coords[image_number][0]
    slice_bound_lat = tiles_coords[image_number][1]

    sub_df = full_df[((ul_lon >= slice_bound_lon[0])) &\
                     ((ul_lon < slice_bound_lon[1])) &\
                     (ul_lat >= slice_bound_lat[0]) &\
                     ((lr_lat < slice_bound_lat[1]))]

    return sub_df



def get_permutations(ndim_list):
    """
    returns all permutations of sublist elements
    """
    if len(ndim_list) == 1:
        res = ndim_list
    else:
        if len(ndim_list) == 2:
            res = [[i, j] for i in ndim_list[0] for j in ndim_list[1]]
        elif len(ndim_list) == 3:
            res = [[i, j, k] for i in ndim_list[0] for j in ndim_list[1] for k in ndim_list[2]]
        else:
            res = [[i, j, k, l] for i in ndim_list[0] for j in ndim_list[1] for k in ndim_list[2] for l in ndim_list[3]]
    return res


def get_correct_odering(list):
    """
    returns correct order of coords for
    Polygon
    """
    p1, p2 = list[2:4]
    list[2], list[3] = p2, p1

    return list

def get_corners(slice_coords: list):
    """
    returs all 4 corners of all tiles
    """

    return [get_correct_odering(get_permutations(coords)) for coords in slice_coords]


if __name__ == '__main__':
    city = 'Paris'
    df_path = os.path.join(INPUT_PATH, city, f'{city}.csv')
    coords_path = os.path.join(INPUT_PATH, city, f'{city}.npy')

    df = pd.read_csv(df_path)
    coords = get_all_sub_coords(df)

    np.save(coords_path, coords)
