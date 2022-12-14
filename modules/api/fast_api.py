from fastapi import FastAPI
import uvicorn

import pandas as pd

from modules.ml_logic.registry import load_model
from modules.ml_logic.preprocessing import preprocess_features
from modules.data_aggregation.params import CITY_BOUNDING_BOXES
from modules.interface.main import pred

app = FastAPI()
app.state.model = load_model()


@app.get("/predict_city")
def predict(city: str  # Paris, London, Berlin Brussels
            ):

    """
    API GET endpoint to obtain heat distribution of given city
    """

    output_dict = pred(city)
    print(output_dict)

    return output_dict


@app.get("/")
def root():
    return {'status': 'ok'}


if __name__ == '__main__':
    uvicorn.run("fast_api:app", host="127.0.0.1", port=8888, reload= True)
