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


def get_all_sub_coords(city: str,
                       full_df: pd.DataFrame,
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

    total_num_tiles = int(len(full_df)/(num_px_lon * num_px_lat))

    # set max longitude and latitude dimensions from original image
    bb_lon_min, bb_lon_max = CITY_BOUNDING_BOXES[city][0][::-1]
    bb_lat_min, bb_lat_max = CITY_BOUNDING_BOXES[city][1][::-1]

    max_lon = int((bb_lon_max- bb_lon_min)/num_px_lon)
    max_lat = int((bb_lat_max- bb_lat_min)/num_px_lat)

    tiles_coords = []

    print(max_lon, max_lat)

    for i in range(0, max_lon, num_px_lon):

        for j in range(0, max_lat, num_px_lat, num_px_lat):

            # get pixel size
            lat_size, lon_size = lr_lat[j] - ul_lat[j], lr_lon[i] - ul_lon[i]

            print(lat_size, lon_size)

            # step size
            step_lat = num_px_lat * lat_size
            step_lon = num_px_lon * lon_size

            # overlap in lon, lat
            lat_overlap = overlap_percent/100 * step_lat
            lon_overlap = overlap_percent/100 * step_lon

            print(lon_overlap, lat_overlap)

            # divide slice_coords into lists of lat and lon
            slice_bound_lat = [ul_lat[j] - lat_overlap, ul_lat[j] + step_lat - lat_overlap]
            slice_bound_lon = [ul_lon[i] - lon_overlap, ul_lon[i] + step_lon - lon_overlap]

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


###################### better implementation ###########################

def get_split_indices(data_array: np.array, number_features: int):
    """
    returns index where array has to be split
    """
    return len(data_array)/number_features


def split_array(data_array: np.array, split_index:int):
    """
    returns split array
    """
    return [data_array[i:i+split_index] for i in range(0,len(data_array), split_index)]


def get_sub_tiles(data: pd.DataFrame, num_px_lon: int, num_px_lat:int):
    """"
    description:
    converts input dataframe to numpy array with shape of lon, lat and depth
    of features

    input:
    data: dataframe, num_px_lon, num_px_lat: required dimensions of subtiles

    returns:
    list of data tensor subtiles with dimensions num_px_lon in longitude and
    num_px_lat in latitude dimension
    and subtile bb coords in [[lon_min, lon_max], [lat_min, lat_max]]
    """

    # reduce size of df
    if 'geometry' in data.columns:
        df_red = data.drop(columns=['LST', 'ele','ll_corner', 'ur_corner', 'lr_corner','bb', 'geometry'])
    else:
        df_red = data.drop(columns=['LST', 'ele','ll_corner', 'ur_corner', 'lr_corner','bb'])

    # convert str to list
    df_red['ul_corner'] = df_red.ul_corner.apply(literal_eval)

    # separate into lat and lon
    df_red['lon'] = np.array([row[0] for row in df_red.ul_corner])[:,0]
    df_red['lat'] = np.array([row[0] for row in df_red.ul_corner])[:,1]

    # set lon, lat as index and unstack
    data_coord_array = df_red.set_index(['lon', 'lat']).unstack().sort_index()
    data_array = data_coord_array.drop(columns = 'ul_corner').values
    coord_array = data_coord_array['ul_corner'].values

    # split data array
    number_features = len(df_red.columns)-3 # minus lat, lon, ul_corner
    split_index_data = int(get_split_indices(data_array[0], number_features))

    # transform data_array
    data_array_trans = np.array([np.array(split_array(array, split_index_data)).T for array in data_array])

    # split coords array
    split_index_coord = int(get_split_indices(coord_array[0], 1))

    # transform coord_array
    coord_array_trans = np.array([np.array(split_array(array, split_index_coord)).T for array in coord_array])

    lon_dim, lat_dim = data_array_trans.shape[:2]

    lon_range = range(0, lon_dim - num_px_lon, num_px_lon)
    lat_range = range(0, lat_dim - num_px_lat, num_px_lat)

    # divide data and coords into subtiles
    data_tiles = np.array([data_array_trans[i:i+num_px_lon, j:j+num_px_lat, :] for i in lon_range for j in lat_range])
    coord_tiles = np.array([coord_array_trans[i:i+num_px_lon, j:j+num_px_lat, :] for i in lon_range for j in lat_range])

    # select just the coord tiles boundaries
    # coord_bb = np.array([[[coords[j,0,0][0][0], coords[-1,-1,0][0][0]], [coords[j,0,0][0][1], coords[-1,-1,0][0][1]]] for coords in coord_tiles\
    #                         for j in range(coord_tiles.shape[1]-1)])

    coord_bb = np.array([[[coords[0,0,0][0][0], coords[-1,-1,0][0][0]], [coords[0,0,0][0][1], coords[-1,-1,0][0][1]]] for coords in coord_tiles])


    return data_tiles, coord_bb


def tile_whole_city(city:str, num_px_lon: int = 32, num_px_lat: int = 32):
    """
    tiles whole city in with size num_px_lon in longitude and
    num_px_lat in latitude dimension
    """

    # import csv data
    data_in_path = os.path.join(INPUT_PATH, city, f'{city}.csv')

    data = pd.read_csv(data_in_path)
    data_tiles, coord_bb = get_sub_tiles(data, num_px_lon, num_px_lat)

    # export both data and coords bounding boxes
    data_ex_path = os.path.join(INPUT_PATH, city, f'{city}_data_tiles_{num_px_lon}_{num_px_lat}.npy')
    coords_ex_path = os.path.join(INPUT_PATH, city, f'{city}_coordbb_tiles_{num_px_lon}_{num_px_lat}.npy')

    np.save(data_ex_path, data_tiles)
    np.save(coords_ex_path, coord_bb)

    return data_tiles, coord_bb


def import_data_array(city: str, num_px_lon: int = 32, num_px_lat: int = 32) -> np.array:
    """
    import data array of subtiles of specific city
    """
    data_path = os.path.join(INPUT_PATH, city, f'{city}_data_tiles_{num_px_lon}_{num_px_lat}.npy')

    return np.load(data_path)


def import_bb_array(city: str, num_px_lon: int = 32, num_px_lat: int = 32) -> np.array:
    """
    import boounding box coordinate array of subtiles of specific city
    """
    data_path = os.path.join(INPUT_PATH, city, f'{city}_coordbb_tiles_{num_px_lon}_{num_px_lat}.npy')

    return np.load(data_path)



if __name__ == '__main__':
    city = 'Paris'
    tile_whole_city(city, num_px_lon= 32, num_px_lat = 32)
