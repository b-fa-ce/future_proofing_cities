import numpy as np
import pandas as pd
from ast import literal_eval
import os
import random

from modules.data_aggregation.params import CITY_BOUNDING_BOXES
from modules.ml_logic.preprocessing import preprocess_features

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
        res = [[i, j] for i in ndim_list[0] for j in ndim_list[1]]

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


def get_subpoints_one(point: list, lon_lat_dist: list, scaling_factor: int = 2):
    """
    returns point, divided by scaling_factor
    """
    diff_lon, diff_lat = lon_lat_dist
    #one point
    corner_diffs = [[[diff_lon/scaling_factor, 0], [0,0], [0, diff_lat/scaling_factor], [diff_lon/scaling_factor, diff_lat/scaling_factor]], # first four subpoints
                    [[diff_lon, 0], [diff_lon/scaling_factor,0], [diff_lon/scaling_factor, diff_lat/scaling_factor], [diff_lon, diff_lat/scaling_factor]], # second
                    [[0,diff_lat/scaling_factor], [0, diff_lat], [diff_lon/scaling_factor, diff_lat], [diff_lon/scaling_factor, diff_lat/scaling_factor]], # 3rd
                    [[diff_lon, diff_lat/scaling_factor], [diff_lon/scaling_factor,diff_lat/scaling_factor], [diff_lon/scaling_factor, diff_lat], [diff_lon, diff_lat]]] # 4th

    sub_points = point + corner_diffs

    return sub_points


def get_all_subpoints(data: pd.DataFrame, scaling_factor:int = 2):
    """
    returns all points divided by four
    """
    # ul, lr corners
    lr_corner = np.array([corner for corner in data['lr_corner'].apply(literal_eval)])
    ul_corner = np.array([corner for corner in data['ul_corner'].apply(literal_eval)])

    # difference between points
    corner_diff_lon_lat = np.array([(lr_corner-ul_corner)[:,0,0], (lr_corner-ul_corner)[:,0,1]]).T

    all_points = [get_subpoints_one(ul_corner[i],corner_diff_lon_lat[i], scaling_factor) for i in range(len(ul_corner))]

    return np.array(all_points).reshape(np.multiply(*np.array(all_points).shape[:2]),4,2)



def subpixels_city(city:str, scaling_factor: int = 2):
    """
    saves subpixels of city to disk
    """
    # import csv data
    data_in_path = os.path.join(INPUT_PATH, city, f'{city}.csv')
    data = pd.read_csv(data_in_path)

    # get subpixels
    subpixels = get_all_subpoints(data)

    # export
    subpixels_ex_path = os.path.join(INPUT_PATH, city, f'{city}_subpixels_{scaling_factor}.npy')

    np.save(subpixels_ex_path, subpixels)


def import_subpixels(city: str, scaling_factor: int = 2) -> np.array:
    """
    import boounding box subpixel array of subtiles of specific city
    """
    data_path = os.path.join(INPUT_PATH, city, f'{city}_subpixels_{scaling_factor}.npy')

    return np.load(data_path)

def get_average_temperature_per_tile(import_data: np.array, index: int = 0) -> np.array:
    """
    returns average temperature difference to mean temp
    per tile for given city
    """
    # import_data = import_data_array(city)

    return np.array([np.mean(tile) for tile in import_data[:,:,:,index]])

def get_zero_buildings_lat_range(df):

    # group by latitude and get mean of building coverage
    build_den = pd.DataFrame(df.groupby(by='lat')['building_coverage'].mean().sort_index()).reset_index()

    # get every averaged lon row close to zero
    build_zero = build_den[build_den.building_coverage <= .005]

    # get the indexes of the pixels that span the latitudes of missing data
    indexes_for_bounds = []
    start_strip = True
    for i in range(len(build_zero.index)):
        # add the beginning of a missing strip of data
        if start_strip:
            indexes_for_bounds.append(build_zero.index[i])
            start_strip = False
            continue
        # if it is the last one, add that+1
        if build_zero.index[i] == build_zero.index[-1]:
            indexes_for_bounds.append(build_zero.index[i]+1)
        # if the next one is not consecutive, then is the next of the missing strip
        elif build_zero.index[i]+1 != build_zero.index[i+1]:
            indexes_for_bounds.append(build_zero.index[i]+1)
            start_strip = True

    # constructing the ranges in pairs
    lat_pairs = []
    start_pair = True
    for index in indexes_for_bounds:
        if start_pair:
            pair = [build_den.iloc[index,:].lat]
            start_pair = False
            continue
        if not start_pair:
            pair.append(build_den.iloc[index,:].lat)
            lat_pairs.append(pair)
            start_pair = True

    return lat_pairs

def get_useful_strips(df):
    df_red = df.drop(columns=['LST', 'ele','ll_corner', 'ur_corner', 'bb'])

    # convert str to list
    df_red['ul_corner'] = df_red.ul_corner.apply(literal_eval)
    df_red['lr_corner'] = df_red.lr_corner.apply(literal_eval)

    # drop irrelevant index
    if 'Unnamed: 0' in df_red.columns:
        df_red.drop(columns='Unnamed: 0', inplace=True)

    # get lon, lat of ul corner
    df_red['lon'] = np.array([row[0] for row in df_red.ul_corner])[:,0]
    df_red['lat'] = np.array([row[0] for row in df_red.ul_corner])[:,1]

    lat_ranges = get_zero_buildings_lat_range(df_red)

    dfs = []
    for i in range(len(lat_ranges)+1):
        if i == 0:
            df = df_red[df_red['lat'] <= lat_ranges[i][0]]
            dfs.append(df_red[df_red['lat'] <= lat_ranges[i][0]])
        elif i == len(lat_ranges):
            dfs.append(df_red[df_red['lat'] >= lat_ranges[i-1][1]])
        else:
            dfs.append(df_red[(df_red['lat'] >= lat_ranges[i-1][1]) & (df_red['lat'] <= lat_ranges[i][0])])

    return dfs

# def old_get_inputs_from_df(df, num_px_lon: int, num_px_lat:int):
#     """
#     Takes a df and returns the tensors that can be fed into the model.
#     First preprocesses, then divides the dataframe into several dataframes
#     consisting of the useful strips of the building data. Then it gets the
#     sub tiles for each of those dataframes and joins them together after.

#     For future reference: The preprocessing is done in parts, which means the
#     standard scaling is done separately for each useful strip. Would be better
#     if they were joined together, scaled as one, then separated again based on
#     the indexes of the original strips.
#     """
#     dfs = get_useful_strips(df)
#     df_preprocessed = []
#     for df in dfs:
#         df_preprocessed.append(preprocess_features(df))

#     data_tensors = []
#     bb_box_tensors = []
#     for df in df_preprocessed:
#         data_tiles, coord_bb = get_sub_tiles(df, num_px_lon, num_px_lat)
#         data_tensors.append(data_tiles)
#         bb_box_tensors.append(coord_bb)

#     tensor_tuple = tuple(tensor for tensor in data_tensors)
#     bb_boxes = tuple(coords for coords in bb_box_tensors)
#     joined_data = np.concatenate((tensor_tuple), axis=0)
#     joined_bb = np.concatenate((bb_boxes), axis=0)

#     return joined_data, joined_bb

def get_inputs_from_df(df, num_px_lon: int, num_px_lat:int):
    """
    ADJUSTED TO DO PREPROCESSING COMBINED

    Takes a df and returns the tensors that can be fed into the model.
    First preprocesses, then divides the dataframe into several dataframes
    consisting of the useful strips of the building data. Then it gets the
    sub tiles for each of those dataframes and joins them together after.
    """
    dfs = get_useful_strips(df)
    lengths = [len(df) for df in dfs]
    merged_df = pd.concat(dfs, ignore_index=False)
    df_preprocessed = preprocess_features(merged_df)
    split_dfs = []
    start = 0
    for i in range(len(lengths)):
        if i == 0:
            split_dfs.append(df_preprocessed.iloc[:lengths[i]])
            start += lengths[i]
        elif i == len(lengths)-1:
            split_dfs.append(df_preprocessed.iloc[start:])
        else:
            split_dfs.append(df_preprocessed.iloc[start:start+lengths[i]])
            start += lengths[i]

    data_tensors = []
    bb_box_tensors = []
    for df in split_dfs:
        data_tiles, coord_bb = get_sub_tiles(df, num_px_lon, num_px_lat)
        data_tensors.append(data_tiles)
        bb_box_tensors.append(coord_bb)

    tensor_tuple = tuple(tensor for tensor in data_tensors)
    bb_boxes = tuple(coords for coords in bb_box_tensors)
    joined_data = np.concatenate((tensor_tuple), axis=0)
    joined_bb = np.concatenate((bb_boxes), axis=0)

    return joined_data, joined_bb


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

    data_array = data.drop(columns= 'lr_corner').set_index(['lon','lat']).unstack().sort_index()
    data_array = data_array.drop(columns = 'ul_corner').values

    coord_array = data[['ul_corner', 'lr_corner', 'lon', 'lat']].set_index(['lon','lat']).unstack().sort_index()
    coord_array_ul = coord_array['ul_corner'].values
    coord_array_lr = coord_array['lr_corner'].values

    coord_array = coord_array_ul

    # split data array
    number_features = len(data.columns)-4 # minus lat, lon, ul_corner, lr_corner
    split_index_data = int(get_split_indices(data_array[0], number_features))

    # transform data_array
    data_array_trans = np.array([np.array(split_array(array, split_index_data)).T for array in data_array])

    # split coords array
    split_index_coord = int(get_split_indices(coord_array[0], 1))

    # transform coord_array
    coord_array_trans_ul = np.array([np.array(split_array(array, split_index_coord)).T for array in coord_array_ul])
    coord_array_trans_lr = np.array([np.array(split_array(array, split_index_coord)).T for array in coord_array_lr])

    lon_dim, lat_dim = data_array_trans.shape[:2]
    lon_range = np.arange(0, lon_dim - num_px_lon, num_px_lon)
    lat_range = np.arange(0, lat_dim - num_px_lat, num_px_lat)

    # divide data and coords into subtiles
    data_tiles = np.array([data_array_trans[i:i+num_px_lon, j:j+num_px_lat, :] for i in lon_range for j in lat_range])
    coord_tiles_ul = np.array([coord_array_trans_ul[i:i+num_px_lon, j:j+num_px_lat, :] for i in lon_range for j in lat_range])
    coord_tiles_lr = np.array([coord_array_trans_lr[i:i+num_px_lon, j:j+num_px_lat, :] for i in lon_range for j in lat_range])

    coord_bb = np.array([[[coord_tiles_ul[i][0,0,0][0][0], coord_tiles_lr[i][-1,-1,0][0][0]], \
                        [coord_tiles_ul[i][0,0,0][0][1], coord_tiles_lr[i][-1,-1,0][0][1]]]\
                            for i in range(len(coord_tiles_ul))])


    return data_tiles, coord_bb

def fill_missing_strips(df):
    df_red = df.drop(columns=['LST', 'ele','ll_corner', 'ur_corner', 'bb'])

    # convert str to list
    df_red['ul_corner'] = df_red.ul_corner.apply(literal_eval)
    df_red['lr_corner'] = df_red.lr_corner.apply(literal_eval)

    # drop irrelevant index
    if 'Unnamed: 0' in df_red.columns:
        df_red.drop(columns='Unnamed: 0', inplace=True)

    # get lon, lat of ul corner
    df_red['lon'] = np.array([row[0] for row in df_red.ul_corner])[:,0]
    df_red['lat'] = np.array([row[0] for row in df_red.ul_corner])[:,1]

    lat_ranges = get_zero_buildings_lat_range(df_red)

    indexes_of_missig = []
    mean_density = df['building_coverage'].mean()
    std_density = df['building_coverage'].std()
    mean_height = df['av_building_height'].mean()
    std_height = df['av_building_height'].std()

    for i in range(len(lat_ranges)):
        indexes_of_empty = (df_red['lat'] > lat_ranges[i][0]) & (df_red['lat'] < lat_ranges[i][1])
        df_red.loc[indexes_of_empty, 'building_coverage'] = random.uniform(0.0, mean_density*2)
        df_red.loc[indexes_of_empty, 'av_building_height'] = random.uniform(mean_height-std_height, mean_height+std_height)

    return df_red

def get_imputed_info_from_df(df, num_px_lon: int, num_px_lat:int):
    """
    Takes a df and returns the tensors that can be fed into the model.
    First preprocesses, then divides the dataframe into several dataframes
    consisting of the useful strips of the building data. Then it gets the
    sub tiles for each of those dataframes and joins them together after.

    For future reference: The preprocessing is done in parts, which means the
    standard scaling is done separately for each useful strip. Would be better
    if they were joined together, scaled as one, then separated again based on
    the indexes of the original strips.
    """
    filled_df = fill_missing_strips(df)
    df_preprocessed = preprocess_features(filled_df)

    data_tiles, coord_bb = get_sub_tiles(df_preprocessed, num_px_lon, num_px_lat)

    return data_tiles, coord_bb

if __name__ == '__main__':
    city = 'Paris'
    # create subtiles
    tile_whole_city(city, num_px_lon= 32, num_px_lat = 32)

    # create subpixels
    subpixels_city(city, 2)
