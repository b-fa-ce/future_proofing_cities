from tensorflow.keras import layers
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.callbacks import EarlyStopping


from sklearn.pipeline import make_pipeline
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.model_selection import cross_val_score, GridSearchCV

from modules.data_aggregation.params import CITY_BOUNDING_BOXES
from modules.ml_logic.utils import slice_picture_coords


# ToDO: perform slicing of full image of 100x100 = 10000 slices -> where best to do it?


def initialize_model(n: tuple) -> Model:
    """
    initialise Neural Network
    n =  n_pixel_lat * n_pixel_lon * number of features
    This model predicts delta T/sub-tile (loss on MSE)
    """
    model = Sequential()

    ### First Convolution & MaxPooling
    model.add(layers.Conv2D(8, (5,5), input_shape=n, padding = 'same', activation = 'relu'))
    model.add(layers.MaxPool2D(pool_size = (2,2)))

    ### Second Convolution & MaxPooling
    model.add(layers.Conv2D(16, (3,3), activation = 'relu'))
    model.add(layers.MaxPool2D(pool_size = (2,2)))

    ### Flattening
    model.add(layers.Flatten())

    ### One Fully Connected layer - "Fully Connected" is equivalent to saying "Dense"
    model.add(layers.Dense(10, activation = 'relu'))

    ### Last layer - Regression layer with one output, the prediction
    model.add(layers.Dense(1, activation = 'linear'))

    ### Model compilation
    model.compile(loss='mse', optimizer = 'adam', metrics = ['mae'])

    return model


def grid_search_params():
    pass

def compile_model():
    pass

# clustering and finding (next to) nearest neighbours on geographical data
# cluster analysis
# output different clusters on map
# grid search? -> model
# kmeans ,gaussian mixture → extra feature?

# clustering on target

# CNN -> slice up into smaller images?
