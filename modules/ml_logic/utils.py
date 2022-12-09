import numpy as np
import pandas as pd
from ast import literal_eval


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


def get_sub_tile(image_data, tiles_coords: list, image_number: int) -> pd.DataFrame:
    """
    returns the sub image according to the smaller
    tile coordinates
    """
    # convert imported upper left corner str into list
    ul_import = image_data.ul_corner.apply(literal_eval)

    # divide ul_corner into lists of lat and lon
    ul_lat = np.array([ul[0] for ul in ul_import])[:,0]
    ul_lon = np.array([ul[0] for ul in ul_import])[:,1]

    # divide slice_coords into lists of lat and lon
    slice_bound_lat = tiles_coords[image_number][0]
    slice_bound_lon = tiles_coords[image_number][1]

    sub_image = image_data[(ul_lat >= slice_bound_lat[0]) &\
                           ((ul_lat < slice_bound_lat[1])) &\
                           ((ul_lon >= slice_bound_lon[0])) &\
                           ((ul_lon < slice_bound_lon[1]))]

    return sub_image
