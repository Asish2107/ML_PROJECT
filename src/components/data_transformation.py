import sys
from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from src.exception import CustomException
from src.logger import logging
import os
from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts', "preprocessor.pkl")

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self):
        try:
            numerical_columns = ['ID', 'Age', 'Tumor Size (cm)', 'Cancer Stage',
                                 'Survival Rate (5-Year, %)', 'Cost of Treatment (USD)',
                                 'Economic Burden (Lost Workdays per Year)']
            categorical_columns = [
                'Country', 'Gender', 'Tobacco Use', 'Alcohol Consumption', 'HPV Infection',
                'Betel Quid Use', 'Chronic Sun Exposure', 'Poor Oral Hygiene',
                'Diet (Fruits & Vegetables Intake)', 'Family History of Cancer',
                'Compromised Immune System', 'Oral Lesions', 'Unexplained Bleeding',
                'Difficulty Swallowing', 'White or Red Patches in Mouth',
                'Treatment Type', 'Early Diagnosis'
            ]

            num_pipeline = Pipeline(steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler())
            ])

            cat_pipeline = Pipeline(steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("one_hot_encoder", OneHotEncoder()),
                ("scaler", StandardScaler(with_mean=False))
            ])

            preprocessor = ColumnTransformer([
                ("num_pipeline", num_pipeline, numerical_columns),
                ("cat_pipeline", cat_pipeline, categorical_columns)
            ])
            return preprocessor

        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_transformation(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            preprocessing_obj = self.get_data_transformer_object()

            target_column = "Oral Cancer (Diagnosis)"
            input_feature_train_df = train_df.drop(columns=[target_column])
            target_feature_train_df = train_df[[target_column]]

            input_feature_test_df = test_df.drop(columns=[target_column])
            target_feature_test_df = test_df[[target_column]]

            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)

            label_encoder = LabelEncoder()
            target_feature_train_df[target_column] = label_encoder.fit_transform(target_feature_train_df[target_column])
            target_feature_test_df[target_column] = label_encoder.transform(target_feature_test_df[target_column])

            train_arr = np.c_[input_feature_train_arr, target_feature_train_df]
            test_arr = np.c_[input_feature_test_arr, target_feature_test_df]

            save_object(self.data_transformation_config.preprocessor_obj_file_path, preprocessing_obj)

            return train_arr, test_arr, self.data_transformation_config.preprocessor_obj_file_path

        except Exception as e:
            raise CustomException(e, sys)
