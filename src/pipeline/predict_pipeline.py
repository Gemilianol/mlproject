import sys
import pandas as pd
from src.exception import CustomException
from src.utils import load_object

class PredictPipeline:
    def __init__(self):
        pass
    
    def predict(self, features):
        try:
        
            model_path='artifacts\model.pkl'
            preprocessor_path='artifacts\preprocessor.pkl'
            model=load_object(file_path=model_path)
            preprocessor=load_object(file_path=preprocessor_path)
            data_scaled=preprocessor.transform(features)
            preds = model.predict(data_scaled)
            return preds
        except Exception as e:
            raise CustomException(e,sys)
    
    
class CustomData:
    def __init__(self, gender:str, lunch: int): #Adapt this line to my case
        self.gender = gender #and so on and so for
        
    def get_data_as_data_frame():
        try:
            custom_data_input_dict = {
                "gender": [self.gender],
                #and so on
            }
            
            return pd.DataFrame(custom_data_input_dict)
        except Exception as e:
            raise CustomException(e,sys)