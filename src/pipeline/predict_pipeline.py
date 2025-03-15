import numpy as np
import os
import sys
import pandas as pd
from src.exception import CustomException
from src.utils import load_object


class PredictPipeline:
    def __init__(self):
        pass

    def predict(self, features):
        try:
            model_path = os.path.join("artifacts", "model.pkl")
            preprocessor_path = os.path.join('artifacts', 'preprocess.pkl')

            model = load_object(file_path=model_path)
            preprocess_dict = load_object(file_path=preprocessor_path)

            scaler = preprocess_dict['scaler']
            label_encoders = preprocess_dict['label_encoders']

            # Columns
            numerical_columns = ['Age', 'Tumor_Size_cm']
            categorical_columns = ['Country', 'Gender', 'Tobacco_Use', 'Alcohol_Consumption', 'Betel_Quid_Use']

            # Rename if needed (in case it's "Tumor_size" in form input)
            if 'Tumor_size' in features.columns:
                features.rename(columns={'Tumor_size': 'Tumor_Size_cm'}, inplace=True)

            # Scale numerical columns
            features[numerical_columns] = scaler.transform(features[numerical_columns])

            # Encode categorical columns
            for col in categorical_columns:
                if col in features.columns:
                    le = label_encoders.get(col)
                    if le is not None:
                        features[col] = le.transform(features[col])
                    else:
                        raise CustomException(f"LabelEncoder not found for column: {col}", sys)

            # Final prediction
            preds = model.predict(features)

            # Decode target if label encoder is available
            target_encoder = label_encoders.get('Oral_Cancer_Diagnosis')
            if target_encoder is not None:
                # Ensure preds is an integer array
                preds = np.round(preds).astype(int)  # Convert float predictions to integer class indices
                preds = preds.reshape(-1)  # Ensure correct shape
                preds = target_encoder.inverse_transform(preds)

            return preds

        except Exception as e:
            raise CustomException(e, sys)


class CustomData:
    def __init__(self,
                 Country: str,
                 Age: int,
                 Gender: str,
                 Tobacco_Use: str,
                 Tumor_size: float,
                 Alcohol_Consumption: str,
                 Betel_Quid_Use: str):

        self.Country = Country
        self.Age = Age
        self.Gender = Gender
        self.Tobacco_Use = Tobacco_Use
        self.Tumor_size = Tumor_size
        self.Alcohol_Consumption = Alcohol_Consumption
        self.Betel_Quid_Use = Betel_Quid_Use

    def get_data_as_data_frame(self):
        try:
            custom_data_input_dict = {
                "Country": [self.Country],
                "Age": [self.Age],
                "Gender": [self.Gender],
                "Tobacco_Use": [self.Tobacco_Use],
                "Tumor_size": [self.Tumor_size],  # Will be renamed to Tumor_Size_cm inside pipeline
                "Alcohol_Consumption": [self.Alcohol_Consumption],
                "Betel_Quid_Use": [self.Betel_Quid_Use]
            }

            return pd.DataFrame(custom_data_input_dict)

        except Exception as e:
            raise CustomException(e, sys)