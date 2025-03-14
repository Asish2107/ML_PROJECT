import sys
import pandas as pd
from src.exception import CustomException
from src.utils import load_object


class PredictPipeline:
    def __init__(self):
        pass

    def predict(self,features):
        try:
            model_path=os.path.join("artifacts","model.pkl")
            preprocessor_path=os.path.join('artifacts','preprocessor.pkl')
            print("Before Loading")
            model=load_object(file_path=model_path)
            preprocessor=load_object(file_path=preprocessor_path)
            print("After Loading")
            data_scaled=preprocessor.transform(features)
            preds=model.predict(data_scaled)
            return preds
        
        except Exception as e:
            raise CustomException(e,sys)



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

        self.Tobacco_Use= Tobacco_Use

        self.Tumor_size = Tumor_size

        self.Alcohol_Consumption= Alcohol_Consumption

        self.Betel_Quid_Use = Betel_Quid_Use

    def get_data_as_data_frame(self):
        try:
            custom_data_input_dict = {
                "Country": [self.Country],
                "Age": [self.Age],
                "Gender": [self.Gender],
                "Tobacco_Use": [self.Tobacco_Use],
                "Tumor_size": [self.Tumor_size],
                "Alcohol_Consumption": [self.Alcohol_Consumption],
                "Betel_Quid_Use": [self.Betel_Quid_Use]
            }

            return pd.DataFrame(custom_data_input_dict)

        except Exception as e:
            raise CustomException(e, sys)
