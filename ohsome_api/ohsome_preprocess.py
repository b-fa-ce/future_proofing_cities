import requests
import pandas as pd
from ohsome import OhsomeClient
import matplotlib.pyplot as plt

#https://wiki.openstreetmap.org/wiki/Key:landuse
def get_overall_landuse_data(bbox:list, time:str, filter:str = 'landuse=*') -> pd.DataFrame:
    '''
    Returns a stacked pd DataFrame with the types of landuse for multiple bboxes.
    The inputs are:
    bbox: is a list of the coordinates of the bbox,
    time: time which we are interested in (format: YY-MM-DD),
    filter: what filter we want to look at most probably landuse=*
    '''
    client = OhsomeClient(log=False)
    groupByKey = 'landuse'
    response = client.elements.count.groupByBoundary.groupByTag.post(bboxes=bbox, time=time, filter=filter, groupByKey=groupByKey)
    return response.as_dataframe()


def get_data_for_map(bbox:list, time:str, filter:str):
    '''
    Returns both the Geopandas dataframe (response_df) and the map of an area based on the landuse
    The inputs are:
    bbox: is a list of the coordinates of the bbox,
    time: time which we are interested in (format: YY-MM-DD),
    filter: what filter we want to look at, here landuse=*
    '''
    client = OhsomeClient(log=False)
    properties = 'tags'
    response = client.elements.geometry.post(bboxes= bbox, time=time, filter= filter, properties=properties)
    response_df = response.as_dataframe()
    fig, ax = plt.subplots(1,1, figsize = (20,20))
    return response_df, response_df.plot(column = 'landuse', ax = ax, legend= True)


def clean_data_for_map(df):
    '''
    The input for this method is the dataframe from the get_map_and_data function and returns a cleaned DataFrame
    All of the columns a part from 'geometry' and 'landuse' will be dropped as the levels of NaNs are very high (min is 87&)
    '''
    return df[['geometry', 'landuse']]

def get_map(df):
    fig, ax = plt.subplots(1,1, figsize = (20,20))
    return df.plot(column = 'landuse', ax=ax, legend = True)
