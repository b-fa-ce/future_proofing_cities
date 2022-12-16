import glob
import os
import time
import pickle

from colorama import Fore, Style

from tensorflow.keras import Model, models


LOCAL_REGISTRY_PATH = os.path.expanduser(os.environ.get("LOCAL_REGISTRY_PATH"))
DOCKER_REGISTRY_PATH = os.path.expanduser(os.environ.get("DOCKER_REGISTRY_PATH"))


def save_model(model: Model = None,
               params: dict = None,
               metrics: dict = None) -> None:
    """
    persist trained model, params and metrics
    """

    timestamp = time.strftime("%Y%m%d-%H%M%S")


    print(Fore.BLUE + "\nSave model to local disk..." + Style.RESET_ALL)


    # check if folder exists and otherwise create it
    if not os.path.exists(LOCAL_REGISTRY_PATH):
        os.makedirs(LOCAL_REGISTRY_PATH)

    # save params
    if params is not None:

        params_path = os.path.join(LOCAL_REGISTRY_PATH, "params")

        # check if folder existsand otherwise create it
        if not os.path.exists(params_path):
           os.makedirs(params_path)

        # save params to file
        params_file_path = os.path.join(params_path, timestamp + ".pickle")
        print(f"- params path: {params_file_path}")
        with open(params_file_path, "wb") as file:
            pickle.dump(params, file)

    # save metrics
    if metrics is not None:

        metrics_path = os.path.join(LOCAL_REGISTRY_PATH, "metrics")

        # check if folder existsand otherwise create it
        if not os.path.exists(metrics_path):
           os.makedirs(metrics_path)

        metrics_file_path = os.path.join(metrics_path, timestamp + ".pickle")
        print(f"- metrics path: {metrics_file_path}")
        with open(metrics_file_path, "wb") as file:
            pickle.dump(metrics, file)

    # save model
    if model is not None:

        model_path = os.path.join(LOCAL_REGISTRY_PATH, "models")

        # check if folder existsand otherwise create it
        if not os.path.exists(model_path):
           os.makedirs(model_path)


        model_file_path = os.path.join(model_path, timestamp)
        print(f"- model path: {model_file_path}")
        model.save(model_file_path)

    print("\n✅ data saved locally")

    return None


def load_model() -> Model:
    """
    load the latest saved model, return None if no model found
    """

    print(Fore.BLUE + "\nLoad model from local disk..." + Style.RESET_ALL)

    # get latest model version
    model_directory = os.path.join(DOCKER_REGISTRY_PATH, "models")

    results = glob.glob(f"{model_directory}/*")
    if not results:
        return None

    model_path = sorted(results)[-1]
    print(f"- path: {model_path}")

    model = models.load_model(model_path)
    print("\n✅ model loaded from disk")

    return model
