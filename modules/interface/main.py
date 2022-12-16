import numpy as np
import pandas as pd
import geopandas
from shapely.geometry import Polygon


from modules.ml_logic.preprocessing import preprocess_features
from modules.ml_logic.utils import get_average_temperature_per_tile, get_corners
from modules.ml_logic.model import initialize_model, compile_model, train_model, evaluate_model
from modules.ml_logic.registry import load_model, save_model
from modules.ml_logic.data import get_data

import os

# General constant variables
TILE_SIZE_LAT = int(os.path.expanduser(os.environ.get("TILE_SIZE_LAT")))
TILE_SIZE_LON = int(os.path.expanduser(os.environ.get("TILE_SIZE_LON")))

LEARNING_RATE = float(os.path.expanduser(os.environ.get("LEARNING_RATE")))
PATIENCE = int(os.path.expanduser(os.environ.get("PATIENCE")))
BATCH_SIZE = int(os.path.expanduser(os.environ.get("BATCH_SIZE")))

PRED_PATH = os.path.expanduser(os.environ.get("PRED_PATH"))



def train(city: str):
    """
    train a new model with preprocessed dataset
    """

    # import data & convert to tensor
    data_array = get_data(city,
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
    data_array =  get_data(city=city.title(),
                           tile_size_lon=TILE_SIZE_LON,
                           tile_size_lat=TILE_SIZE_LAT,
                           context='train')[0]



    # features and target
    X = data_array[:,:,:,1:].astype('float64')
    y = get_average_temperature_per_tile(data_array, 0)

    # load model
    model = load_model()

    if model == None:
        print("\n❌ Model non-existent -> train model first")
        raise

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



def pred(city: str) -> np.ndarray:
    """
    Make a prediction using the trained model and saving
    the resulting geojson to disk
    """

    # output path for predicted data
    pred_path_city = os.path.join(PRED_PATH,city)

    if not os.path.exists(pred_path_city):
        os.makedirs(pred_path_city)

    # prediction data
    data_pred, bb_pred = get_data(city=city.title(), tile_size_lon=TILE_SIZE_LON, tile_size_lat=TILE_SIZE_LAT, context='predict')
    X_pred = data_pred[:,:,:,1:].astype('float64')

    if X_pred is None:
        print('Please specify data')

    # load model
    model = load_model()

    if model == None:
        print("\n❌ Model non-existent -> train model first")
        raise

    # predict
    y_pred = model.predict(X_pred)

    print("\n✅ prediction done: number of tiles", y_pred.shape)

    # create gdf
    pred_gdf = geopandas.GeoDataFrame(y_pred, geometry = [Polygon(get_corners([bb])[0]) for bb in bb_pred])
    pred_gdf.columns = ['LST_diff', 'geometry']

    # add centre points
    pred_gdf['centre_points'] = [pred_gdf.geometry[i].centroid.wkt for i in np.arange(len(pred_gdf))]

    # generalise file path
    pred_path_city_gdf = os.path.join(pred_path_city, f'{city}_viz.geojson')
    pred_gdf.to_file(pred_path_city_gdf, driver='GeoJSON')

    # return y_pred
    return  {'city': city, 'mean_temperature_difference': float(np.mean(y_pred)), 'output_path': PRED_PATH, 'gdf': pred_gdf.to_json()}


if __name__ == '__main__':
    # train model on Paris
    train("Paris")
