from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder
from sklearn.pipeline import make_pipeline
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.model_selection import cross_val_score, GridSearchCV

from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping


import pandas as pd
import numpy as np


## basic structure

# features: av building height/pix, building density/pix,
# landcover types per pix (12 categories), elevation (difference to mean elevation)/pix
# target: difference to mean temp/pix


# preprocessing
def create_features_preprocessor() -> ColumnTransformer:
    """
    create prepocessor for input features
    1. av building height -> StandardScaler() (Normal distribution) (, MinMaxScaler())
    2. density: fine
    3. elevation: MinMaxScaler() -> look at distribution
    4. landcover: OHE
    and returns preprocessed numpy array
    """

    # instantiate scalers
    standard = StandardScaler()
    minmax = MinMaxScaler()

    # instantiate ohe
    ohe = OneHotEncoder(handle_unknown='ignore')


    # COMBINED PREPROCESSOR
    preprocessor = ColumnTransformer(
        [
            ("building_height_scaler", standard, ['building_height']),
            ("elevation_scaler", minmax, ['ele']),
            ("landcover_ohe", ohe, ['categories']),
        ],
        n_jobs=-1,
    )

    return preprocessor


def train_preprocess_features(X: pd.DataFrame):
    """
    returns fitted preprocessor
    use on X_train only
    """
    # instantiate preproecessor
    preprocessor = create_features_preprocessor()
    preprocessor.fit(X)

    return preprocessor

def preprocess_features(X: pd.DataFrame, preprocessor) -> np.array:
    """
    returns preprocessed features
    use on X_train, X_val and X_test separately
    """

    return preprocessor.transform(X)


if __name__ == '__main__':
    preprocessor = create_features_preprocessor()
    X_processed = preprocessor.fit_transform(X)




# features:
# 1. av building height -> StandardScaler() (Normal distribution) (, MinMaxScaler())
# 2. density: fine
# 3. elevation: MinMaxScaler() -> look at distribution
# 4. landcover: OHE

# target: fine



# clustering and finding (next to) nearest neighbours on geographical data
# cluster analysis
# output different clusters on map
# grid search? -> model
# kmeans ,gaussian mixture → extra feature?

# clustering on target

# CNN -> slice up into smaller images?
