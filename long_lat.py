import numpy as np
import math
import pyproj
from functools import partial
from shapely.ops import transform
from shapely.geometry import Point
from pyproj import Proj
from shapely.geometry import shape
from operator import itemgetter

def Haversine(lat1,lon1,lat2,lon2, **kwarg):
    """
    This uses the ‘haversine’ formula to calculate the great-circle distance between two points – that is,
    the shortest distance over the earth’s surface – giving an ‘as-the-crow-flies’ distance between the points
    (ignoring any hills they fly over, of course!).
    Haversine
    formula:    a = sin²(Δφ/2) + cos φ1 ⋅ cos φ2 ⋅ sin²(Δλ/2)
    c = 2 ⋅ atan2( √a, √(1−a) )
    d = R ⋅ c
    where   φ is latitude, λ is longitude, R is earth’s radius (mean radius = 6,371km);
    note that angles need to be in radians to pass to trig functions!
    """
    R = 6371.0088
    lat1,lon1,lat2,lon2 = map(np.radians, [lat1,lon1,lat2,lon2])

    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2) **2
    c = 2 * np.arctan2(a**0.5, (1-a)**0.5)
    d = R * c
    return round(d,4)


def geodesic_point_buffer(lat, lon, km):
    proj_wgs84 = pyproj.Proj('+proj=longlat +datum=WGS84')
    # Azimuthal equidistant projection
    aeqd_proj = '+proj=aeqd +lat_0={lat} +lon_0={lon} +x_0=0 +y_0=0'
    project = partial(
        pyproj.transform,
        pyproj.Proj(aeqd_proj.format(lat=lat, lon=lon)),
        proj_wgs84)
    buf = Point(0, 0).buffer(km * 1000)  # distance in metres
    b = transform(project, buf).exterior.coords[:]
    north_dot = max(b,key=itemgetter(0))
    south_dot = min(b,key=itemgetter(0))
    east_dot = max(b,key=itemgetter(1))
    west_dot = min(b,key=itemgetter(1))

    return north_dot, south_dot, east_dot, west_dot


def geodesic_point_buffer_vector(lat, lon, km):
    north_dot, south_dot, east_dot, west_dot = geodesic_point_buffer(lat,lon,km)
    north_vector = np.array(north_dot) - np.array([lat, lon])
    south_vector = np.array(south_dot) - np.array([lat, lon])
    east_vector = np.array(east_dot) - np.array([lat, lon])
    west_vector = np.array(west_dot) - np.array([lat, lon])

    return north_vector, south_vector, east_vector, west_vector

def new_pixel(in_pixel, dist):
    """
    input: pixel: ll, ul, ur, lr
    padding: distance
    """
    # ll
    lon, lat = in_pixel[0]
    north_vector, south_vector, east_vector, west_vector = geodesic_point_buffer_vector(lat, lon, dist)
    ll_new = np.array(in_pixel[0]) + np.array(south_vector) + np.array(west_vector)
    # ul
    lon, lat = in_pixel[1]
    north_vector, south_vector, east_vector, west_vector = geodesic_point_buffer_vector(lat, lon, dist)
    ul_new = np.array(in_pixel[1]) + north_vector + west_vector
    # ur
    lon, lat = in_pixel[2]
    north_vector, south_vector, east_vector, west_vector = geodesic_point_buffer_vector(lat, lon, dist)
    ur_new = np.array(in_pixel[2]) + north_vector + east_vector
    # lr
    lon, lat = in_pixel[3]
    north_vector, south_vector, east_vector, west_vector = geodesic_point_buffer_vector(lat, lon, dist)
    lr_new = np.array(in_pixel[3]) + south_vector + east_vector
    return [ll_new, ul_new, ur_new, lr_new]


def calc_area(lis_lats_lons):

    lons, lats = zip(*lis_lats_lons)
    ll = list(set(lats))[::-1]
    var = []
    for i in range(len(ll)):
        var.append('lat_' + str(i+1))
    st = ""
    for v, l in zip(var,ll):
        st = st + str(v) + "=" + str(l) +" "+ "+"
    st = st +"lat_0="+ str(np.mean(ll)) + " "+ "+" + "lon_0" +"=" + str(np.mean(lons))
    tx = "+proj=aea +" + st
    pa = Proj(tx)

    x, y = pa(lons, lats)
    cop = {"type": "Polygon", "coordinates": [zip(x, y)]}

    return shape(cop).area
