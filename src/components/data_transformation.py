import sys
from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, StandardScaler
from imblearn.over_sampling import SMOTE  # Added for class imbalance handling
from src.exception import CustomException
from src.logger import logging
import os
from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts', "preprocess.pkl")

class DataTransformation:
    def __init__(self):
        self.config = DataTransformationConfig()

    def initiate_data_transformation(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            target_column = "Oral_Cancer_Diagnosis"
            numerical_columns = ['Age', 'Tumor_Size_cm']
            categorical_columns = ['Country', 'Gender', 'Tobacco_Use', 'Alcohol_Consumption', 'Betel_Quid_Use']

            # === Check Class Imbalance
            class_distribution = train_df[target_column].value_counts()
            logging.info(f"Class Distribution in Training Set: \n{class_distribution}")

            # === Split input/target
            X_train = train_df.drop(columns=[target_column]).copy()
            y_train = train_df[target_column].copy()
            X_test = test_df.drop(columns=[target_column]).copy()
            y_test = test_df[target_column].copy()

            # === Impute numerical
            num_imputer = SimpleImputer(strategy='median')
            X_train[numerical_columns] = num_imputer.fit_transform(X_train[numerical_columns])
            X_test[numerical_columns] = num_imputer.transform(X_test[numerical_columns])

            # === Scale numerical
            scaler = StandardScaler()
            X_train[numerical_columns] = scaler.fit_transform(X_train[numerical_columns])
            X_test[numerical_columns] = scaler.transform(X_test[numerical_columns])

            # === Encode categorical
            label_encoders = {}
            for col in categorical_columns:
                cat_imputer = SimpleImputer(strategy='most_frequent')
                X_train[col] = cat_imputer.fit_transform(X_train[[col]]).ravel()
                X_test[col] = cat_imputer.transform(X_test[[col]]).ravel()

                le = LabelEncoder()
                X_train[col] = le.fit_transform(X_train[col])
                X_test[col] = le.transform(X_test[col])
                label_encoders[col] = le

            # === Encode target
            target_le = LabelEncoder()
            y_train = target_le.fit_transform(y_train)
            y_test = target_le.transform(y_test)
            label_encoders[target_column] = target_le

            # === Handle Class Imbalance using SMOTE
            smote = SMOTE(random_state=42)
            X_train, y_train = smote.fit_resample(X_train, y_train)
            logging.info(f"After SMOTE: {np.bincount(y_train)}")  # Log class distribution

            # === Final arrays
            train_arr = np.c_[X_train.values, y_train.reshape(-1, 1)]
            test_arr = np.c_[X_test.values, y_test.reshape(-1, 1)]

            # === Save ONE combined preprocessor object
            preprocess_obj = {
                'scaler': scaler,
                'label_encoders': label_encoders
            }
            save_object(self.config.preprocessor_obj_file_path, preprocess_obj)

            logging.info("Data transformation complete. Preprocessor saved successfully.")
            return train_arr, test_arr, self.config.preprocessor_obj_file_path

        except Exception as e:
            raise CustomException(e, sys)