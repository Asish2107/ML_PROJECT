import os
import sys
import pickle
import dill
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV

from src.exception import CustomException

def save_object(file_path, obj):
    """
    Save the given object to a file using pickle.

    Parameters:
    - file_path (str): The path to save the file.
    - obj (object): The object to be saved.
    """
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)


def evaluate_models(X_train, y_train, X_test, y_test, models: dict):
    """
    Evaluate multiple models using GridSearchCV and return a report of test accuracy scores.

    Parameters:
    - X_train: Training features
    - y_train: Training labels
    - X_test: Test features
    - y_test: Test labels
    - models (dict): Dictionary of models. Keys should match keys in param.
    - param (dict): Dictionary of parameter grids. Each value should contain a 'model' and 'params' key.

    Returns:
    - report (dict): Dictionary containing the best test accuracy scores for each model.
    """
    try:
        report = {}

        for model_name in models.keys():
            model = models[model_name]
            # model_params = param[model_name]['params']

            # # Grid Search for best hyperparameters
            # gs = GridSearchCV(model, model_params, cv=3, n_jobs=-1, verbose=0)
            # gs.fit(X_train, y_train)

            best_model = gs.best_estimator_
            best_model.fit(X_train, y_train)

            y_train_pred = best_model.predict(X_train)
            y_test_pred = best_model.predict(X_test)

            train_score = accuracy_score(y_train, y_train_pred) * 100
            test_score = accuracy_score(y_test, y_test_pred) * 100

            report[model_name] = test_score  # You can log train_score if needed too

        return report

    except Exception as e:
        raise CustomException(e, sys)


def load_object(file_path):
    """
    Load an object from a pickle file.

    Parameters:
    - file_path (str): Path of the pickle file

    Returns:
    - The loaded Python object
    """
    try:
        with open(file_path, "rb") as file_obj:
            return pickle.load(file_obj)

    except Exception as e:
        raise CustomException(e, sys)
