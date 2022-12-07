import numpy as np
import matplotlib.pyplot as plt
import mpmath as mp
import requests




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


def get_split_coords(building):
    """Separate lon and lat
    """
    coords = building['geometry']['coordinates'][0][:]
    x = []
    y = []
    for coord in coords:
        x.append(coord[0])
        y.append(coord[1])
    return x, y

def plot_json(json):
    """
    plot buildings in json
    """
    for building in json['features']:
        x, y = get_split_coords(building)
    plt.plot(x, y)

def plot_whole_json_set(json_set):
    """
    plots all buildings in a dict of jsons
    """
    for json in json_set.values():
        plot_json(json)
    plt.show;

def receive_building_data(jsons):
  buildings = []
  for json in jsons.values():
    for building in json['features']:
      height = building['properties']['height']
      coords = building['geometry']['coordinates'][0]
      building_data = [height, coords]
      buildings.append(building_data)
  return buildings


def polyarea(x,y):
    return 0.5*np.abs(np.dot(x,np.roll(y,1))-np.dot(y,np.roll(x,1)))
