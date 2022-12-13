from modules.ml_logic.utils import get_sub_tiles
from modules.ml_logic.preprocessing import preprocess_features

import pandas as pd
import os

INPUT_PATH = "../../data_processed"


def get_data(city:str,
             preprocess: bool = True,
            #  context: str = 'train',
             tile_size_lon: int = 6,
             tile_size_lat: int = 6) -> list:
    """
    imports data for given city and returns list of
    sub-tiled data-array and bounding box coords
    """

    path = os.path.join(INPUT_PATH,city,'_full.csv')
    df = pd.read_csv(path).drop(columns = "Unnamed: 0")

    # preprocess features
    if preprocess:
        prep_df = preprocess_features(df)
    else:
        prep_df = df

    # divide into tiles and convert to numpy array
    return get_sub_tiles(prep_df, tile_size_lon, tile_size_lat)
