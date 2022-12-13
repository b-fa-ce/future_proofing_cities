import streamlit as st
import json
import requests
import geopandas
import pyproj
import plotly.graph_objs as go
import pandas as pd
from folium.plugins import HeatMap
import folium
from streamlit_folium import st_folium
import fiona
from shapely.geometry import shape
import numpy as np


# data = requests.get("'https://jsonplaceholder.typicode.com/todos/1'").json()
# st.write(data)
df = geopandas.read_file(filename = "Paris_viz.geojson")
data_n = df[['LST_diff', 'centre_points']]
data_new = data_n[['centre_points', 'LST_diff']]
data_new['centre_points'] = data_new.centre_points.apply(lambda y: np.array([float(x) for x in y[7:-1].split()]))
data_new_2 = data_new.apply(lambda x: [x['centre_points'][1], x['centre_points'][0], x['LST_diff']], axis=1)
data_new_2_upd = list(data_new_2.to_numpy())

st.title('Probable heat islands')
st.text('This is a web app to allow find heat islands in different cities')
option = st.selectbox("Type input city", ['Paris', 'Brussels', 'London', 'Berlin'])


point_array  = data_new_2_upd

if option == 'Paris':
    # lat = 48.864716
    # lon = 2.349014
    m = folium.Map(location=[48.864716, 2.349014], tiles = 'openstreetmap', radius=100, zoom_start=10, max_val=5.0, control_scale=True)
    HeatMap(point_array).add_to(m)
elif option == 'Brussels':
    # lat = 50.8476
    # lon = 4.3572
    m = folium.Map(location=[50.8476, 4.3572], tiles = 'openstreetmap', zoom_start=10, max_val=5.0, control_scale=True)
    HeatMap(point_array).add_to(m)
elif option == 'London':
    # lat = 51.5072
    # lon = 0.1276
    m = folium.Map(location=[51.5072, 0.1276], tiles = 'openstreetmap', zoom_start=10, max_val=5.0, control_scale=True)
    HeatMap(point_array).add_to(m)
else:
    # lat = 52.5200
    # lon = 13.4050
    m = folium.Map(location=[52.5200, 13.4050], tiles = 'openstreetmap', zoom_start=10, max_val=5.0, control_scale=True)
    HeatMap(point_array).add_to(m)

st_folium(m)













# map_data = pd.DataFrame({'lat': [lat], 'lon': [lon]})
# st.map(map_data)

# Get lat and lon coordinates of a tile
# ll_tiles['lon'] = points["geometry"].x
# ll_tiles['lat'] = points["geometry"].y

# ul_tiles['lon'] = points["geometry"].x
# ul_tiles['lat'] = points["geometry"].y

# ur_tiles['lon'] = points["geometry"].x
# ur_tiles['lat'] = points["geometry"].y

# lr_tiles['lon'] = points["geometry"].x
# lr_tiles['lat'] = points["geometry"].y

# tiles_array = points[['lat', 'lon']].values

#
