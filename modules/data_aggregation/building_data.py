import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mpmath as mp
import requests
from shapely.geometry import Polygon
from long_lat import pad_pixel


def get_tile(lat_deg, lon_deg, zoom=15):
    """
    A function to get the relevant tile from lat,lon,zoom)
    """
    lat_rad = mp.radians(lat_deg)
    n = 2 ** zoom
    xtile = n * ((lon_deg + 180) / 360)
    ytile = float(n * (1 - (mp.log(mp.tan(lat_rad) + mp.sec(lat_rad)) / np.pi)) / 2)
    return zoom, round(xtile), round(ytile) # 'tile %d/%d/%d '%

def tile2lon(z,x,y):
    return x / 2**z * 360 - 180

def tile2lat(z,x,y):
    n = mp.pi - 2 * mp.pi * y / 2**z;
    return float((180 / mp.pi) * (mp.atan(0.5 * (mp.exp(n) - mp.exp(-n)))))

def tile_bbox(z,x,y):
    '''
    Returns the lat, lon bounding box of a tile
    '''
    w = tile2lon(z,x,y)
    s = tile2lat(z,x,y)
    e = tile2lon(z,x+1,y)
    n = tile2lat(z,x,y+1)
    return [w,s,e,n]

def osmbuildings_request(latitude:float, longitude:float):
    """
    returns json response with building data from OSMBuildings
    API for a specific latitude and longitude
    """
    base_url = "https://data.osmbuildings.org/0.2/anonymous/tile"
    zoom, xtile, ytile = get_tile(latitude, longitude, 15)
    url = f"{base_url}/{zoom}/{xtile}/{ytile}.json"
    print(f"URL: {url}")
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; rv:91.0) Gecko/20100101 Firefox/91.0'}
    response = requests.get(url, headers = headers)
    json_response = response.json()
    return json_response

def get_tiles(city_coords):
    """
    Gets the number of tiles that form the grid over the city
    """
    ul_tile = get_tile(lat_deg=city_coords['upper_left'][0], lon_deg=city_coords['upper_left'][1], zoom=15)
    lr_tile = get_tile(lat_deg=city_coords['lower_right'][0], lon_deg=city_coords['lower_right'][1], zoom=15)
    city_xtiles = np.abs(ul_tile[1] - lr_tile[1])
    city_ytiles = np.abs(ul_tile[2] - lr_tile[2])
    city_tiles = [city_xtiles, city_ytiles]
    return city_tiles

def get_all_tiles(city_coords, city_tiles):
    """
    Returns two lists, one for each x and one for each y
    """
    starting_tile = []
    starting_tile.append(get_tile(city_coords['upper_left'][0], city_coords['upper_left'][1])[1])
    starting_tile.append(get_tile(city_coords['upper_left'][0], city_coords['upper_left'][1])[2])
    x_tiles = []
    y_tiles = []
    for x in range(city_tiles[0]-1):
        x_tiles.append(starting_tile[0] + x)
    for y in range(city_tiles[1]-1):
        y_tiles.append(starting_tile[1] + y)
    return x_tiles, y_tiles

def get_all_tile_jsons(x_tiles, y_tiles, zoom=15):
    """
    Returns jsons for all combinations of x and y within a given city
    """
    city_jsons = {}
    for x in x_tiles:
        for y in y_tiles:
            tile_lat = tile2lat(zoom, x, y)
            tile_lon = tile2lon(zoom, x, y)
            tile = (x, y)
            json_response = osmbuildings_request(tile_lat, tile_lon)
            city_jsons[tile] = json_response
    return city_jsons

def receive_building_data(jsons):
    """
    Splits the building jsons to receive only the useful parts
    """
    buildings = []
    for json in jsons.values():
        for building in json['features']:
            height = building['properties']['height']
            coords = building['geometry']['coordinates'][0]
            building_data = [height, coords]
            buildings.append(building_data)
    return buildings

def get_split_coords(coords):
    """Separate lon and lat
    """
    x = []
    y = []
    for coord in coords:
        x.append(coord[0])
        y.append(coord[1])
    return x, y

def plot_pixel_and_polygons(polygons, pixel):
    """
    Plots a pixel and the polygons on same axis
    """
    pixel_returning = pixel.copy()
    pixel_returning.append(pixel[0])
    x, y = get_split_coords(pixel_returning)
    plt.plot(x, y)
    for polygon in polygons:
        x, y = get_split_coords(polygon)
        plt.plot(x, y)
    return

def visualise_buildings_pixel(pixels, buildings, number, clip):
    """
    Combines functions so when given a list of pixels and the index,
    it plots out that pixel, all polygons, and gives a percentage coverage
    """
    chosen_pixel = pixels[number]
    matching_buidlings = get_pixel_buildings(buildings, chosen_pixel)
    plot_pixel_and_polygons(matching_buidlings, chosen_pixel)
    coverage = get_building_coverage(buildings, clip, chosen_pixel)
    plt.title(f"Building coverage is {round(coverage, 3)}%.")
    plt.show();

def visualise_landuse_pixel(pixels, landuse_polygons, number, clip):
    """
    Combines functions so when given a list of pixels and the index,
    it plots out that pixel, all landuse polygons, and gives a percentage coverage.
    For testing/verifying only.
    """
    chosen_pixel = pixels[number]
    matching_buidlings = get_pixel_polygons(landuse_polygons, chosen_pixel)
    plot_pixel_and_polygons(matching_buidlings, chosen_pixel)
    coverage = get_landuse_coverage(landuse_polygons, clip, chosen_pixel)
    plt.title(f"Landuse coverage is {round(coverage, 3)}%.")
    plt.show();

def get_pixel_polygons(polygons, pixel):
    """
    Returns the polygons that have a point inside a given pixel, mostly just for
    testing and visualising the logic.
    """
    local_polygons = []
    for polygon in polygons:
        for point in polygon:
            if is_polygon_point_inside(polygon, pixel):
                local_polygons.append(polygon)
                continue
    return local_polygons

def get_pixel_buildings(buildings, pixel):
    """
    Returns the buildings that have a point inside a given pixel, mostly just for
    testing and visualising the logic.
    """
    local_buildings = []
    for building in buildings:
        coords = building[1]
        for point in coords:
            if is_polygon_inside(coords, pixel):
                local_buildings.append(coords)
                continue
            elif is_polygon_point_inside(coords, pixel):
                local_buildings.append(coords)
                continue
    return local_buildings

def get_clipped_polygon_area(polygon_coords, clip, pixel):
    """
    Given a clipper class and coordinates for an intersecting polygon & pixel,
    returns the area of the clipped polygon
    """
    clipped = clip(polygon_coords, pixel)
    pgon = Polygon(clipped)
    area = pgon.area
    return area

def is_polygon_inside(polygon_coords, pixel):
    """
    Checks if all points of a polygon are inside a pixel
    """
    inside=True
    for coord in polygon_coords:
        if pixel[0][0] < coord[0] < pixel[2][0] and pixel[0][1] < coord[1] < pixel[1][1]:
            pass
        else:
            inside = False
    return inside

def is_polygon_point_inside(polygon_coords, pixel):
    """
    Checks if any point of a polygon is inside a pixel
    """
    inside=False
    for coord in polygon_coords:
        if pixel[0][0] < coord[0] < pixel[2][0] and pixel[0][1] < coord[1] < pixel[1][1]:
            inside = True
        else:
            continue
    return inside

# def get_building_coverage(buildings, clip, pixel):
#     """
#     Gets the percentage coverage of buildings in a pixel, used for looping over
#     all the pixels. Requires buildings to be saved from the return_building_data
#     function defined above, and an instantiated clipper class
#     """
#     separate_areas = []
#     for building in buildings:
#         building_coords = building[1]
#         if is_polygon_inside(building_coords, pixel):
#             pgon = Polygon(building_coords)
#             separate_areas.append(pgon.area)
#         elif is_polygon_point_inside(building_coords, pixel):
#             area = get_clipped_polygon_area(building_coords, clip, pixel)
#             separate_areas.append(area)
#     built_area = np.sum(separate_areas)
#     pixel_poly = Polygon(pixel)
#     pixel_area = pixel_poly.area
#     prop_area_built = built_area/pixel_area
#     return prop_area_built

# def get_landuse_coverage(landuses, clip, pixel):
#     """
#     Returns the percentage of a landuse category for each pixel. Needs to be passed
#     an instantiated clipping class, a series of the polygons from one landuse category,
#     and one pixel. Used for iterating over each pixel.
#     """
#     separate_areas = []
#     for landuse_coords in landuses:
#         if is_polygon_inside(landuse_coords, pixel):
#             pgon = Polygon(landuse_coords)
#             separate_areas.append(pgon.area)
#         elif is_polygon_point_inside(landuse_coords, pixel):
#             area = get_clipped_polygon_area(landuse_coords, clip, pixel)
#             separate_areas.append(area)
#     area_covered = np.sum(separate_areas)
#     pixel_poly = Polygon(pixel)
#     pixel_area = pixel_poly.area
#     prop_area_covered = area_covered/pixel_area
#     return prop_area_covered

def get_full_landuse_coverage(landuse_polygons, clip, pixel, padding_distance):
    """
    Returns the proportional coverage if a landuse category for each pixel.
    Needs to be passed an instantiated clipping class (MODIFIED CLIPPER ONLY),
    a series of the polygons from one landuse category, one pixel,
    and the distance in metres to pad the pixel by in each dimension.
    Used for iterating over each pixel.
    """
    separate_areas = []
    local_polygons = []
    padded_pixel = pad_pixel(pixel, padding_distance)
    for landuse_polygon in landuse_polygons:
        if is_polygon_point_inside(landuse_coords, padded_pixel):
            local_polygons.append(landuse_polygon)
    for landuse_coords in local_polygons:
        if is_polygon_inside(landuse_coords, pixel):
            pgon = Polygon(landuse_coords)
            separate_areas.append(pgon.area)
        elif is_polygon_point_inside(landuse_coords, pixel):
            area = get_clipped_polygon_area(landuse_coords, clip, pixel)
            separate_areas.append(area)
        else:
            area = get_clipped_polygon_area(landuse_coords, clip, pixel)
            if area == False:
                continue
            separate_areas.append(area)
    area_covered = np.sum(separate_areas)
    pixel_poly = Polygon(pixel)
    pixel_area = pixel_poly.area
    prop_area_covered = area_covered/pixel_area
    return prop_area_covered

# def get_building_coverage_and_height(buildings, clip, pixel):
#     """
#     Gets the percentage coverage of buildings in a pixel,
#     alongside the average height
#     """
#     separate_areas = []
#     heights = []
#     for building in buildings:
#         building_coords = building[1]
#         if is_polygon_inside(building_coords, pixel):
#             pgon = Polygon(building_coords)
#             separate_areas.append(pgon.area)
#             heights.append(building[0])
#         elif is_polygon_point_inside(building_coords, pixel):
#             area = get_clipped_polygon_area(building_coords, clip, pixel)
#             separate_areas.append(area)
#             heights.append(building[0])
#     built_area = np.sum(separate_areas)
#     proportional_areas = [separate_area/built_area for separate_area in separate_areas]
#     proportional_heights = [proportional_areas[i] * heights[i] for i in range(len(proportional_areas))]
#     average_height = np.sum(proportional_heights)
#     pixel_poly = Polygon(pixel)
#     pixel_area = pixel_poly.area
#     prop_area_built = built_area/pixel_area
#     return prop_area_built, average_height

def get_full_coverage_and_height(buildings, clip, pixel, padding_distance):
    """
    Returns the proportional building coverage and the averaged height for each pixel.
    Needs to be passed an instantiated clipping class (MODIFIED CLIPPER ONLY),
    a series of the polygons from one landuse category, one pixel,
    and the distance in metres to pad the pixel by in each dimension.
    Used for iterating over each pixel.
    """
    separate_areas = []
    heights = []
    local_buildings = []
    padded_pixel = pad_pixel(pixel, padding_distance)
    for building in buildings:
        building_coords = building[1]
        if is_polygon_point_inside(building_coords, padded_pixel):
            local_buildings.append(building)
    for building in local_buildings:
        building_coords = building[1]
        if is_polygon_inside(building_coords, pixel):
            pgon = Polygon(building_coords)
            separate_areas.append(pgon.area)
            heights.append(building[0])
        elif is_polygon_point_inside(building_coords, pixel):
            area = get_clipped_polygon_area(building_coords, clip, pixel)
            separate_areas.append(area)
            heights.append(building[0])
        else:
            area = get_clipped_polygon_area(building_coords, clip, pixel)
            if area == False:
                continue
            separate_areas.append(area)
            heights.append(building[0])
    built_area = np.sum(separate_areas)
    proportional_areas = [separate_area/built_area for separate_area in separate_areas]
    proportional_heights = [proportional_areas[i] * heights[i] for i in range(len(proportional_areas))]
    average_height = np.sum(proportional_heights)
    pixel_poly = Polygon(pixel)
    pixel_area = pixel_poly.area
    prop_area_built = built_area/pixel_area
    return prop_area_built, average_height

def get_pd_series_full_coverage_height(row):
    """

  

    This functions requires all building data to be saved as 'paris_buildings',
    and a clipper saved as clip (USE MODIFIED CLIPPER),
    and can only be used on the paris dataframe from paris.csv when the bb has had
    literal_eval applied and saved to a new column bounds.
    Very specific function, would need to be tweaked to work for anything else.

    """
    prop_area_built, pixel_average_height = get_full_coverage_and_height(paris_buildings, clip, row['bounds'])
    return pd.Series([prop_area_built, pixel_average_height])