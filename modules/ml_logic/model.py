from colorama import Fore, Style

import numpy as np
from typing import Tuple

from tensorflow.keras import layers
from tensorflow.keras import Sequential, Model, optimizers
from tensorflow.keras.callbacks import EarlyStopping





def initialize_model(in_shape: tuple) -> Model:
    """
    initialise Neural Network
    in_shape =  (n_pixel_lat, n_pixel_lon, number of features)
    This model predicts delta T/sub-tile (loss on MSE)
    """
    model = Sequential()

    ### First Convolution & MaxPooling
    model.add(layers.Conv2D(filters = 8, kernel_size= (3,3), input_shape=in_shape, padding = 'same', activation = 'relu'))
    # model.add(layers.MaxPool2D(pool_size = (3, 3)))
    model.add(layers.Dropout(0.2))

    ### Second Convolution & MaxPooling
    model.add(layers.Conv2D(16, (2,2), activation = 'relu'))
    model.add(layers.Dropout(0.5))


    ### Second Convolution & MaxPooling
    model.add(layers.Conv2D(32, (2,2), activation = 'relu'))
    model.add(layers.Dropout(0.5))


    ### Flattening
    model.add(layers.Flatten())

    ### One Fully Connected layer - "Fully Connected" is equivalent to saying "Dense"
    model.add(layers.Dense(10, activation = 'relu'))
    model.add(layers.Dropout(0.5))


    ### Last layer - Regression layer with one output, the prediction
    model.add(layers.Dense(1, activation = 'linear'))



    print("\n✅ model initialised")

    return model



def compile_model(model: Model, learning_rate: float = 0.001) -> Model:
    """
    compiles model
    """

    optimizer = optimizers.Adam(learning_rate=learning_rate)
    model.compile(loss="huber", optimizer=optimizer, metrics=["mae", 'mse'])

    print("\n✅ model compiled")
    return model


def train_model(model: Model,
                X: np.ndarray,
                y: np.ndarray,
                batch_size=16,
                patience=15,
                validation_split=0.3,
                validation_data=None) -> Tuple[Model, dict]:
    """
    Fit model and return a the tuple (fitted_model, history)
    """

    print(Fore.BLUE + "\nTrain model..." + Style.RESET_ALL)

    es = EarlyStopping(monitor="val_loss",
                       patience=patience,
                       restore_best_weights=True,
                       verbose=0)

    history = model.fit(X,
                        y,
                        validation_split=validation_split,
                        validation_data=validation_data,
                        epochs=200,
                        batch_size=batch_size,
                        callbacks=[es],
                        verbose=0)

    print(f"\n✅ model trained ({len(X)} rows)")

    return model, history


def evaluate_model(model: Model,
                   X: np.ndarray,
                   y: np.ndarray,
                   batch_size=64) -> Tuple[Model, dict]:
    """
    Evaluate trained model performance on dataset
    """

    print(Fore.BLUE + f"\nEvaluate model on {len(X)} rows..." + Style.RESET_ALL)

    if model is None:
        print(f"\n❌ no model to evaluate")
        return None

    metrics = model.evaluate(
        x=X,
        y=y,
        batch_size=batch_size,
        verbose=1,
        # callbacks=None,
        return_dict=True)

    loss = metrics["loss"]
    mae = metrics["mae"]
    mse = metrics['mse']
    mae_baseline = np.sum(np.abs(y))/len(y)
    mse_baseline = np.sum(y**2)/len(y)

    print(f"\n✅ model evaluated: loss {round(loss, 2)}, mae {round(mae, 2)}, mse: {mse}, baseline mae: {mae_baseline}, baseline mse: {mse_baseline}")

    return metrics
