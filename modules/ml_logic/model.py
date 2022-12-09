from tensorflow.keras import layers
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.callbacks import EarlyStopping


from sklearn.pipeline import make_pipeline
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.model_selection import cross_val_score, GridSearchCV

from modules.data_aggregation.params import CITY_BOUNDING_BOXES
from modules.ml_logic.utils import slice_picture_coords


# ToDO: perform slicing of full image of 100x100 = 10000 slices -> where best to do it?


def initialize_model() -> Model:
    """
    initialise Neural Network
    """
    model = Sequential()

    ### First Convolution & MaxPooling
    model.add(layers.Conv2D(8, (4,4), input_shape=X_train.shape[1:].as_list(), padding = 'same', activation = 'relu'))
    model.add(layers.MaxPool2D(pool_size = (2,2)))

    ### Second Convolution & MaxPooling
    model.add(layers.Conv2D(16, (3,3), activation = 'relu'))
    model.add(layers.MaxPool2D(pool_size = (2,2)))

    ### Flattening
    model.add(layers.Flatten())

    ### One Fully Connected layer - "Fully Connected" is equivalent to saying "Dense"
    model.add(layers.Dense(10, activation = 'relu'))

    ### Last layer - Classification Layer with 10 outputs corresponding to 10 digits
    model.add(layers.Dense(10, activation = 'softmax'))

    ### Model compilation
    model.compile(loss='categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])

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
