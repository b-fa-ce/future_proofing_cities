import numpy as np
import pandas as pd


from modules.ml_logic.preprocessing import preprocess_features
from modules.ml_logic.utils import get_average_temperature_per_tile
from modules.ml_logic.model import initialize_model, compile_model, train_model, evaluate_model
from modules.ml_logic.registry import load_model, save_model
from modules.ml_logic.data import get_data

import os

# General constant variables
TILE_SIZE_LAT = os.path.expanduser(os.environ.get("TILE_SIZE_LAT"))
TILE_SIZE_LON = os.path.expanduser(os.environ.get("TILE_SIZE_LON"))



def train(city: str):
    """
    train a new model with preprocessed dataset
    """

    # General constant variables
    LEARNING_RATE = 0.002
    PATIENCE = 10
    BATCH_SIZE = 25

    # import data & convert to tensor
    data_array = get_data(city,
                          preprocess = True,
                          tile_size_lon = TILE_SIZE_LON,
                          tile_size_lat = TILE_SIZE_LAT)[0]

    # features and target
    X = data_array[:,:,:,1:].astype('float64')
    y = get_average_temperature_per_tile(data_array, 0)

    # initialize model
    model = initialize_model(X.shape[1:])

    # compile model
    model = compile_model(model, learning_rate = LEARNING_RATE)

    # train model
    model, history = train_model(
        model,
        X,
        y,
        batch_size = BATCH_SIZE,
        patience = PATIENCE)


    # Return the min value of the validation MAE
    val_mae = np.min(history.history['val_mae'])

    print(f"\n✅ trained on {X.shape[0]} subtiles with MAE: {round(val_mae, 2)}")

    params = dict(
        # Model parameters
        learning_rate = LEARNING_RATE,
        batch_size = BATCH_SIZE,
        patience = PATIENCE
    )

    # Save model
    save_model(model=model, params=params, metrics=dict(mae=val_mae))

    return val_mae


def evaluate(city: pd.DataFrame):
    """
    Evaluate the performance of the latest production model on new data
    """
    # import data & convert to tensor
    data_array = get_data(city,
                          preprocess = True,
                          tile_size_lon = TILE_SIZE_LON,
                          tile_size_lat = TILE_SIZE_LAT)[0]

    # features and target
    X = data_array[:,:,:,1:].astype('float64')
    y = get_average_temperature_per_tile(data_array, 0)


    model = load_model()

    metrics_dict = evaluate_model(model=model, X=X, y=y)
    mae = metrics_dict["mae"]

    # Save evaluation
    params = dict(
        # Package behavior
        context="evaluate",

        # Data source
        row_count=len(X)
    )

    save_model(params=params, metrics=dict(mae=mae))

    return mae



def pred(X_pred: pd.DataFrame = None) -> np.ndarray:
    """
    Make a prediction using the trained model
    """

    if X_pred is None:
        print('Please specify data')

    model = load_model()

    X_processed = preprocess_features(X_pred)

    y_pred = model.predict(X_processed)

    print("\n✅ prediction done: ", y_pred, y_pred.shape)

    return y_pred


if __name__ == '__main__':
    # train model on Paris
    train("Paris")
    # how to train test split??

    # load new data
    data_array = get_data(city,
                        preprocess = True,
                        tile_size_lon = TILE_SIZE_LON,
                        tile_size_lat = TILE_SIZE_LAT)[0]
