from modules.ml_logic.utils import get_inputs_from_df

import pandas as pd
import os

INPUT_PATH = "../../data/processed_data"


def get_data(city:str,
            #  context: str = 'train',
             tile_size_lon: int = 6,
             tile_size_lat: int = 6) -> list:
    """
    imports data for given city and returns list of
    sub-tiled data-array and bounding box coords
    """

    path = os.path.join(INPUT_PATH,city,f'{city}_full.csv')
    df = pd.read_csv(path)

    # divide into tiles and convert to numpy array
    return get_inputs_from_df(df, tile_size_lon, tile_size_lat)
