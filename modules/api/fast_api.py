from fastapi import FastAPI
import uvicorn

import pandas as pd

from modules.ml_logic.registry import load_model
from modules.ml_logic.preprocessing import preprocess_features
from modules.data_aggregation.params import CITY_BOUNDING_BOXES
from modules.ml_logic.data import get_data

app = FastAPI()
# app.state.model = load_model()

# General constant variables
TILE_SIZE_LAT = 6 # size of 6 pxs
TILE_SIZE_LON = 6 # size of 6 pxs


@app.get("/predict_city")
def predict(city: str  # Paris, London, Berlin Brussels
            ):

    """
    API GET endpoint to obtain heat distribution of given city
    """

    # data, bb = get_data(city,
    #                     preprocess= True,
    #                     tile_size_lon =  TILE_SIZE_LON,
    #                     tile_size_lat =  TILE_SIZE_LAT)

    # X_pred = data[:,:,:,1:].astype('float64')

    # temp_pred = app.state.model.predict(X_pred)

    return {'temp': city}# temp_pred}


@app.get("/")
def root():
    return {'status': 'ok'}


if __name__ == '__main__':
    uvicorn.run("fast_api:app", host="127.0.0.1", port=8888, reload= True)
