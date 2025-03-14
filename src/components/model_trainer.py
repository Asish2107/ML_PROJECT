import os
import sys
from dataclasses import dataclass

from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from catboost import CatBoostClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, evaluate_models


@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts", "model.pkl")


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Splitting training and test input data")
            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1]
            )

            # Define models
            models = {
                "Logistic Regression": LogisticRegression(),
                "K-Nearest Neighbors": KNeighborsClassifier(),
                "Decision Tree": DecisionTreeClassifier(),
                "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
                "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, random_state=42),
                "Support Vector Machine": SVC(probability=True),
                "XGBClassifier": XGBClassifier(),
                "CatBoosting Classifier": CatBoostClassifier(verbose=False),
                "AdaBoostClassifier": AdaBoostClassifier(),
                "Naive Bayes": GaussianNB()
            }

            # Hyperparameter grid for each model
            # param = {
            #     "Logistic Regression": {
            #         "model": LogisticRegression(max_iter=1000),
            #         "params": {
            #             'penalty': ['l1', 'l2'],                          # Type of regularization
            #             'C': [0.001, 0.01, 0.1, 1, 10, 100],              # Regularization strength
            #             'solver': ['liblinear'],                         # Solver used for optimization
            #             'class_weight': [None, 'balanced']               # Handle class imbalance
            #         }
            #     },
            #     "K-Nearest Neighbors": {
            #         "model": KNeighborsClassifier(),
            #         "params": {
            #             'n_neighbors': [3, 5, 7, 9],                     # Number of neighbors to consider
            #             'weights': ['uniform', 'distance'],             # Weighting method
            #             'algorithm': ['auto', 'ball_tree', 'kd_tree'],  # Algorithm for computing nearest neighbors
            #             'leaf_size': [15, 30, 45],                       # Leaf size for tree-based methods
            #             'p': [1, 2]                                      # Power parameter: 1=Manhattan, 2=Euclidean
            #         }
            #     },
            #     "Decision Tree": {
            #         "model": DecisionTreeClassifier(random_state=42),
            #         "params": {
            #             'max_depth': [None, 5, 10, 20],                  # Max depth of tree
            #             'min_samples_split': [2, 5, 10],                # Min samples required to split
            #             'min_samples_leaf': [1, 2, 4],                  # Min samples in leaf node
            #             'criterion': ['gini', 'entropy','poissoin'],              # Impurity measure
            #             'splitter': ['best', 'random']                 # Node split strategy
            #         }
            #     },
            #     "Random Forest": {
            #         "model": RandomForestClassifier(n_estimators=100, random_state=42),
            #         "params": {
            #             'max_depth': [None, 10, 20],
            #             'min_samples_split': [2, 5],
            #             'min_samples_leaf': [1, 2],
            #             'max_features': ['sqrt', 'log2'],              # Number of features to consider at split
            #             'bootstrap': [True, False]                     # Whether bootstrap samples are used
            #         }
            #     },
            #     "Gradient Boosting": {
            #         "model": GradientBoostingClassifier(n_estimators=100, random_state=42),
            #         "params": {
            #             'learning_rate': [0.01, 0.1, 0.2],              # Shrinks contribution of each tree
            #             'max_depth': [3, 5, 7],
            #             'min_samples_split': [2, 5],
            #             'min_samples_leaf': [1, 2],
            #             'subsample': [0.8, 1.0]                         # Fraction of samples used for fitting
            #         }
            #     },
            #     "Support Vector Machine": {
            #         "model": SVC(probability=True, random_state=42),
            #         "params": {
            #             'C': [0.1, 1, 10],                              # Regularization parameter
            #             'kernel': ['linear', 'rbf', 'poly'],           # Kernel type
            #             'gamma': ['scale', 'auto'],                    # Kernel coefficient
            #             'degree': [2, 3, 4],                           # Degree for poly kernel
            #             'class_weight': [None, 'balanced']
            #         }
            #     },
            #     "XGBClassifier": {
            #         "model": XGBClassifier(random_state=42),
            #         "params": {
            #             'learning_rate': [0.01, 0.1],
            #             'max_depth': [3, 5, 7],
            #             'subsample': [0.8, 1.0],
            #             'colsample_bytree': [0.8, 1.0],                # Features used per tree
            #             'n_estimators': [50, 100],
            #             'gamma': [0, 0.1, 0.2]                         # Minimum loss reduction for split
            #         }
            #     },
            #     "CatBoosting Classifier": {
            #         "model": CatBoostClassifier(verbose=False, random_state=42),
            #         "params": {
            #             'depth': [4, 6, 8],                            # Depth of tree
            #             'learning_rate': [0.01, 0.1],
            #             'iterations': [100, 200],                     # Total boosting rounds
            #             'l2_leaf_reg': [1, 3, 5]                       # L2 regularization
            #         }
            #     },
            #     "AdaBoostClassifier": {
            #         "model": AdaBoostClassifier(random_state=42),
            #         "params": {
            #             'n_estimators': [50, 100, 200],
            #             'learning_rate': [0.01, 0.1, 1.0],
            #             'algorithm': ['SAMME', 'SAMME.R']              # SAMME.R is faster and better for probabilistic estimates
            #         }
            #     },
            #     "Naive Bayes": {
            #         "model": GaussianNB(),
            #         "params": {
            #             'var_smoothing': [1e-9, 1e-8, 1e-7]             # Stability parameter for numerical precision
            #         }
            #     }
            # }

            # Evaluate models
            model_report: dict = evaluate_models(X_train, y_train,X_test, y_test,models)

            # Get best model from report
            best_model_score = max(sorted(model_report.values()))
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            best_model = models[best_model_name]

            if best_model_score < 90:
                raise CustomException("No best model found with acceptable score.")

            logging.info(f"Best found model on both training and testing dataset: {best_model_name}")

            # Save best model object
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            # Predict and evaluate on test set
            predicted = best_model.predict(X_test)
            accuracy = accuracy_score(y_test, predicted)
            return accuracy

        except Exception as e:
            raise CustomException(e, sys)
