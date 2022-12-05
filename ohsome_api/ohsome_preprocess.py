import requests
import pandas as pd

#https://wiki.openstreetmap.org/wiki/Key:landuse
def get_data(bbox):
    '''
    Returns a pd DataFrame with the types of landuse for a given city. The input is the name of the city and the associated BBox
    '''
    landuse = ['commercial', 'construction', 'education', 'industrial', 'residential',
               'retail', 'institutional', 'aquaculture', 'allotments', 'farmland',
               'farmyard', 'flowerbed', 'forest', 'greenhouse_horticulture',
               'meadow', 'orchard', 'plant_nursery', 'vineyard', 'basin', 'salt_pond',
               'brownfield', 'cemetery', 'depot', 'garages', 'grass', 'greenfield',
               'landfill', 'military', 'port', 'quarry', 'railway', 'recreation_ground',
               'religious', 'village_green', 'winter_sports']
    URL = 'https://api.ohsome.org/v1/elements/area'
    result = {}
    for element in landuse:
        data = {"bboxes": bbox, "format": "json", "time": "2022-08-01", "filter": f"landuse={element}"}
        response = requests.post(URL, data=data)
        result[element] = response.json().get('result')[0].get('value')
        print("-" * 30)
        print(f"Gathered data for landuse = {element}")
        print("-" *30)
    return result
