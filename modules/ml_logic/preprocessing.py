from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.compose import ColumnTransformer

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
    4. landcover: see if we have to scale it, but should be normalised to [0,1]
    and returns preprocessed numpy array
    """

    # instantiate scalers
    standard = StandardScaler()
    minmax = MinMaxScaler()

    # COMBINED PREPROCESSOR
    preprocessor = ColumnTransformer(
        [
            ("building_height_scaler", standard, ['building_height']),
            ("elevation_scaler", minmax, ['ele']),
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


# if __name__ == '__main__':
#     preprocessor = create_features_preprocessor()
#     X_processed = preprocessor.fit_transform(X)
