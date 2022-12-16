from fastapi import FastAPI
import uvicorn
from fastapi.middleware.cors import CORSMiddleware

import pandas as pd

from modules.ml_logic.registry import load_model
from modules.ml_logic.preprocessing import preprocess_features
from modules.data_aggregation.params import CITY_BOUNDING_BOXES
from modules.interface.main import pred

app = FastAPI()


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

@app.get("/predict_city")
def predict(city: str):  # Paris, London, Berlin Brussels
    """
    API GET endpoint to obtain heat distribution of given city
    """

    output_dict = pred(city)

    return output_dict


@app.get("/")
def root():
    return {'status': 'ok'}


if __name__ == '__main__':
    uvicorn.run("fast_api:app", host="127.0.0.1", port=8000, reload= True)
