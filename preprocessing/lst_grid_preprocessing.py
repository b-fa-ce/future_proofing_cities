import numpy as np
import geopandas
from pyresample import geometry as geom
from pyresample import kd_tree as kdt
import pyproj

from shapely.geometry import Polygon


from preprocessing.lst_preprocessing import import_city_data, \
    convert_to_Celsius, CITY_BOUNDING_BOXES

import os

# input and output file paths
INPUT_PATH = os.path.join('..','raw_data')
OUTPUT_PATH = os.path.join('..','processed_data')


def swath_to_grid(lst_data: np.array,
                  height_data: np.array,
                  lon_data: np.array,
                  lat_data: np.array):
    """
    converts satellite swath to regular grid
    based upon: https://git.earthdata.nasa.gov/projects/LPDUR/repos/tutorial-ecostress/browse/ECOSTRESS_Tutorial.ipynb
    """
    # pixel size is 70m
    PIXEL_SIZE = 70

    # Set swath definition from lat/lon arrays
    swath_def= geom.SwathDefinition(lons=lon_data, lats=lat_data)

    #Define the coordinates in the middle of the swath, which are used
    # to calculate an estimate of the output rows/columns for the gridded output.
    mid = [int(lat_data.shape[1] / 2) - 1, int(lat_data.shape[0] / 2) - 1]
    mid_lat, mid_lon = lat_data[mid[0]][mid[1]], lon_data[mid[0]][mid[1]]

    # Define AEQD projection centered at swath center
    epsg_convert = pyproj.Proj("+proj=aeqd +lat_0={} +lon_0={}".format(mid_lat, mid_lon))

    # Use info from AEQD projection bbox to calculate output cols/rows/pixel size
    ll_lon, ll_lat = epsg_convert(np.min(lon_data), np.min(lat_data), inverse=False)
    ur_lon, ur_lat = epsg_convert(np.max(lon_data), np.max(lat_data), inverse=False)
    area_extent = (ll_lon, ll_lat, ur_lon, ur_lat)

    cols = int(round((area_extent[2] - area_extent[0]) / PIXEL_SIZE))  # 70 m pixel size
    rows = int(round((area_extent[3] - area_extent[1]) / PIXEL_SIZE))

    #Use number of rows and columns generated above from the AEQD projection
    # to set a representative number of rows and columns in the Geographic area definition,
    # which will then be translated to degrees below, then take the smaller
    # of the two pixel dims to determine output size and ensure square pixels.

    # Define Geographic projection
    epsg, proj, proj_name = '4326', 'longlat', 'Geographic'

    # Define bounding box of swath
    ll_lon, ll_lat, ur_lon, ur_lat = np.min(lon_data), np.min(lat_data), np.max(lon_data), np.max(lat_data)
    area_extent = (ll_lon, ll_lat, ur_lon, ur_lat)

    # Create area definition with estimated number of columns and rows
    proj_dict = pyproj.CRS("epsg:4326")
    area_def = geom.AreaDefinition(epsg, proj_name, proj, proj_dict, cols, rows, area_extent)

    # Square pixels and calculate output cols/rows
    pixel_size = np.min([area_def.pixel_size_x, area_def.pixel_size_y])
    cols = int(round((area_extent[2] - area_extent[0]) / pixel_size))
    rows = int(round((area_extent[3] - area_extent[1]) / pixel_size))

    # Set up a new Geographic area definition with the refined cols/rows
    area_def = geom.AreaDefinition(epsg, proj_name, proj, proj_dict, cols, rows, area_extent)

    # Get arrays with information about the nearest neighbor to each grid point
    # function is comparing the swath and area definitions
    # to locate the nearest neighbor (neighbours=1).
    # 210 is the radius_of_influence, or the radius used to search for the nearest
    # neighboring pixel in the swath (in meters).
    index, outdex, index_array, dist_array = kdt.get_neighbour_info(swath_def, area_def, 210, neighbours=1)


    # The second step is to use those arrays to retrieve a resampled result.
    # Perform K-D Tree nearest neighbor resampling (swath 2 grid conversion)
    lst_grid = kdt.get_sample_from_neighbour_info('nn', area_def.shape, lst_data, index,\
                                                  outdex, index_array, fill_value=0)
    height_grid = kdt.get_sample_from_neighbour_info('nn', area_def.shape, height_data, index,\
                                                  outdex, index_array, fill_value=0)

    # Define the geotransform: lower_left latitude, longitude, pixel_size
    geo_transform = [area_def.area_extent[0], area_def.area_extent[1], pixel_size]

    return lst_grid, height_grid, geo_transform



def get_pixel_coords(row_col: list, geo_transform: list) -> list:
    """
    returs list of corner coordinates lat, lon of pixel
    for specific row and column
    format: [upper_left, lower_left, lower_right, upper_right]
    """
    pixel_size = geo_transform[-1]
    row = row_col[0]
    col = row_col[1]

    current_row = geo_transform[1] + pixel_size * (row - 1)
    next_row = current_row + pixel_size

    current_col = geo_transform[0] + pixel_size * (col - 1)
    next_col = current_col + pixel_size

    upper_left = [current_col, current_row]
    lower_left = [current_col, next_row]
    lower_right = [next_col, next_row]
    upper_right = [next_col, current_row]

    return [upper_left, lower_left, lower_right, upper_right]


def coord_bounds_to_pixels(geo_transform, bounds):
    """
    converts coordinate bounds in lon, lat format
    to pixel bounds in lon, lat format
    """
    pixel_size = geo_transform[-1]

    lon_start = round((bounds[0][0]-geo_transform[0])/pixel_size)
    lon_end = round((bounds[0][1]-geo_transform[0])/pixel_size)

    lat_start = round((bounds[1][0]-geo_transform[1])/pixel_size)
    lat_end = round((bounds[1][1]-geo_transform[1])/pixel_size)

    return lon_start,lon_end + 1, lat_start, lat_end + 1



def create_gdf(city):
    """
    build GeoDataFrame for specific city contained in CITY_BOUNDING_BOXES
    """
    lst_data, lat_data, lon_data, height_data = import_city_data(city)

    # gridded data: LST, height, geo_transform
    lst_data_grid, height_data_grid, geo_transform = \
        swath_to_grid(lst_data, height_data, lon_data, lat_data)

    # pixel bounds
    lon_bounds = coord_bounds_to_pixels(geo_transform, CITY_BOUNDING_BOXES[city])[:2]
    lat_bounds = coord_bounds_to_pixels(geo_transform, CITY_BOUNDING_BOXES[city])[2:]

    # pixel ranges
    lon_range = np.arange(*lon_bounds)
    lat_range = np.arange(*lat_bounds)

    # slice data
    lst_slice = lst_data_grid[lat_bounds[0]:lat_bounds[1], lon_bounds[0]:lon_bounds[1]]
    height_slice = height_data_grid[lat_bounds[0]:lat_bounds[1], lon_bounds[0]:lon_bounds[1]]

    # corner coordinates
    corners = np.array([[get_pixel_coords((row, col), geo_transform) \
        for col in lon_range] for row in lat_range])


    # flatten
    lst_flat = lst_slice.reshape(np.multiply(*lst_slice.shape),)
    height_flat = height_slice.reshape(np.multiply(*height_slice.shape),)
    corners_flat = corners.reshape(np.multiply(*corners.shape[:2]),*corners.shape[2:])

    # convert LST to degree Celcius
    lst_flat = convert_to_Celsius(lst_flat)

    # difference to average values
    lst_diff = lst_flat - np.average(lst_flat[lst_flat!=-1000])
    height_diff = height_flat - np.average(height_flat[lst_flat!=-1000])

    # build shapely polygons
    polygons = [Polygon(single_corners) for single_corners in corners_flat]

    # individual corners
    ul_corners = [[list(corner)] for corner in corners_flat[:,0,:]]
    ll_corners = [[list(corner)] for corner in corners_flat[:,1,:]]
    lr_corners = [[list(corner)] for corner in corners_flat[:,2,:]]
    ur_corners = [[list(corner)] for corner in corners_flat[:,3,:]]

    # GeoDataFrame
    gpd = geopandas.GeoDataFrame({'LST': lst_flat.T,
                                 'elevation': height_flat.T,
                                 'LST_diff': lst_diff,
                                 'elevation_diff': height_diff,
                                 'ul_corner': ul_corners,
                                 'll_corner': ll_corners,
                                 'lr_corner': lr_corners,
                                 'ur_corner': ur_corners},
                                  geometry = polygons, crs = "EPSG:4326")

    return gpd


def save_gdf_to_csv(city: str):
    """
    saves DataFrames to csv files
    """

    gdf = create_gdf(city)

    out_path = os.path.join(OUTPUT_PATH,f'{city}.csv')
    gdf.to_csv(out_path, index=False)
    print(f"Saved data for {city} to {os.path.join(OUTPUT_PATH,f'{city}.csv')}")


if __name__ == '__main__':
    for city in CITY_BOUNDING_BOXES:
        print(f"Processing data for {city}...")
        save_gdf_to_csv(city)
