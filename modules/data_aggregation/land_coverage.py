
import geopandas as gpd
import matplotlib.pyplot as plt


def get_data():
    '''
    From the preprocessed_data directory read the .geojson file as a geopandas dataframe
    '''
    gpd_paris = gpd.read_file("../../data/processed_data/Paris/gpd_paris.geojson")
    gpd_paris['geometry_coordinates'] = [list(gpd_paris.geometry[index].exterior.coords)[::-1] for index, row in gpd_paris.iterrows()]
    return gpd_paris


def get_map(gpd):
    '''
    Returns the mapped data of the geopandas dataframe
    '''
    map = gpd.plot(column = 'categories', figsize = (20,20), legend = True, cmap = 'viridis')
    return map
