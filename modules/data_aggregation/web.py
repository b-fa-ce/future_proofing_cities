import streamlit as st
import json
import requests
import geopandas
import pyproj
import plotly.graph_objs as go
import pandas as pd
from folium import plugins
from folium.plugins import HeatMap
import folium
from streamlit_folium import st_folium
import fiona
from shapely.geometry import shape
import numpy as np
from shapely.geometry import Polygon
from branca.colormap import linear

st.title('Probable heat islands')
st.text('This is a web app to allow find heat islands in different cities')
option = st.selectbox("Type input city", ['Paris', 'Brussels', 'London', 'Berlin'])
df = geopandas.read_file(filename = "Paris_viz (1).geojson")

if option == 'Paris':
    m = folium.Map(location=[48.864716, 2.349014], zoom_start=10, tiles='CartoDB positron')
elif option == 'Brussels':
    m = folium.Map(location=[50.8476, 4.3572], tiles = 'CartoDB positron', zoom_start=10, max_val=5.0, control_scale=True)
elif option == 'London':
    m = folium.Map(location=[51.5072, 0.1276], tiles = 'CartoDB positron', zoom_start=10, max_val=5.0, control_scale=True)
else:
    m = folium.Map(location=[52.5200, 13.4050], tiles = 'CartoDB positron', zoom_start=10, max_val=5.0, control_scale=True)

def map_color(heat):
    if -0.95 < heat < -0.5:
        return '#076cf5'
    elif -0.5 <= heat < 0:
        return '#4d97fa'
    elif 0 <= heat < 0.5:
        return '#fcce58'
    elif 0.5 <= heat < 1:
         return '#fca558'
    elif 1 <= heat < 1.5:
         return '#fa6220'
    else:
        return '#f70505'

for _, r in df.iterrows():
    sim_geo = geopandas.GeoSeries(r['geometry']).simplify(tolerance=0.002)
    heat_value = r['LST_diff']
    geo_j = sim_geo.to_json()
    geo_j = folium.GeoJson(data=geo_j,
                           style_function=lambda x,
                           heat_value=heat_value: {'fillColor': map_color(heat_value)})
    folium.Popup(r['LST_diff']).add_to(geo_j)
    geo_j.add_to(m)

count_colormap = linear.RdBu_09.scale(min(df['LST_diff']),
                                            max(df['LST_diff']))
count_colormap.add_to(m)
plugins.Fullscreen(position='topright').add_to(m)
st_folium(m)
# m = folium.Map(location=[48.864716, 2.349014], zoom_start=10, tiles='CartoDB positron')
