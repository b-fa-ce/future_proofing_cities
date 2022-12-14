from colorama import Fore, Style

from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
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
    robust = RobustScaler()
    minmax = MinMaxScaler()

    # COMBINED PREPROCESSOR
    preprocessor = ColumnTransformer(
        [
            ("building_height_scaler", robust, ['av_building_height']),
            ("building_density_scaler", robust, ['building_coverage']),
            ("elevation_scaler", minmax, ['ele_diff']),
        ],
        remainder='passthrough',
        n_jobs=-1,
    )

    return preprocessor


def preprocess_features(X: pd.DataFrame):
    """
    returns fitted preprocessor
    use on X_train only
    """
    print(Fore.BLUE + "\nPreprocess features..." + Style.RESET_ALL)

    # instantiate preproecessor
    preprocessor = create_features_preprocessor()
    X_processed = preprocessor.fit_transform(X)

    col_names = preprocessor.get_feature_names_out()
    col_names = [name.removeprefix("remainder__").removeprefix('elevation_scaler__') for name in col_names]
    col_names = [name.removeprefix("building_density_scaler__").removeprefix('building_height_scaler__') for name in col_names]

    print(col_names)

    df_processed = pd.DataFrame(X_processed, columns=col_names)

    # shift column 'Name' to first position
    first_column = df_processed.pop('LST_diff')

    # insert column using insert(position,column_name,
    # first_column) function
    df_processed.insert(0, 'LST_diff', first_column)

    return df_processed


def preprocess_features_transform_only(X: pd.DataFrame, preprocessor) -> np.array:
    """
    returns preprocessed features
    use on X_train, X_val and X_test separately
    """
    print(Fore.BLUE + "\nPreprocess features..." + Style.RESET_ALL)

    X_processed = preprocessor.transform(X)

    print("\n✅ X_processed, with shape", X_processed.shape)

    return X_processed


# if __name__ == '__main__':
#     preprocessor = create_features_preprocessor()
#     X_processed = preprocessor.fit_transform(X)
