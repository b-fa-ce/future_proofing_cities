import requests
import pandas as pd
from ohsome import OhsomeClient
import matplotlib.pyplot as plt

#https://wiki.openstreetmap.org/wiki/Key:landuse
def get_data(bbox:list, time:str, filter:str = 'landuse=*') -> pd.DataFrame:
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


def get_map(bbox:list, time:str, filter:str):
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
