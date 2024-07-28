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
    def __init__(self, Fats:float, SFats: float, MFats: float, PFats:float,
                 Carbs: float, Sugars:float, Protein:float,
                 DFiber: float, Cholesterol: float, Sodium: float, Water: float,
                 VA: float, VB1: float,VB2: float,VB3: float,VB5: float,VB6: float,
                 VB11: float, VB12: float,VC: float, VD: float, VE: float,VK: float,
                 Calcium: float, Copper: float, Iron: float, Magnesium: float,Manganese: float,
                 Phosphorus: float, Potassium: float, Selenium: float, Zinc: float, cv: float): 
        
        self.fats = Fats
        self.sfats = SFats
        self.mfats = MFats
        self.pfats = PFats
        self.carbs = Carbs
        self.sugars = Sugars
        self.protein = Protein
        self.dfiber = DFiber
        self.cholesterol = Cholesterol
        self.sodium = Sodium
        self.water = Water
        self.va = VA
        self.vb1 = VB1
        self.vb2 = VB2
        self.vb3 = VB3
        self.vb5 = VB5
        self.vb6 = VB6
        self.vb11 = VB11
        self.vb12 = VB12
        self.vc = VC
        self.vd = VD
        self.ve = VE
        self.vk = VK
        self.calcium = Calcium
        self.cooper = Copper
        self.iron = Iron
        self.mag = Magnesium
        self.magna = Manganese
        self.ph = Phosphorus
        self.po = Potassium
        self.sel = Selenium
        self.zinc = Zinc
        self.cv = cv
        
    def get_data_as_data_frame(self):
        try:
            custom_data_input_dict = {
                'Fat': [self.fats],
                'Saturated Fats': [self.sfats],
                'Monounsaturated Fats': [self.mfats],
                'Polyunsaturated Fats': [self.pfats],
                'Carbohydrates':[self.carbs],
                'Sugars': [self.sugars],
                'Protein': [self.protein],
                'Dietary Fiber': [self.dfiber],
                'Cholesterol': [self.cholesterol],
                'Sodium': [self.sodium],
                'Water': [self.water],
                'Vitamin A': [self.va],
                'Vitamin B1': [self.vb1],
                'Vitamin B11': [self.vb11],
                'Vitamin B12': [self.vb12],
                'Vitamin B2': [self.vb2],
                'Vitamin B3': [self.vb3],
                'Vitamin B5': [self.vb5],
                'Vitamin B6': [self.vb6],
                'Vitamin C': [self.vc],
                'Vitamin D': [self.vd],
                'Vitamin E': [self.ve],
                'Vitamin K': [self.vk],
                'Calcium': [self.calcium],
                'Copper': [self.cooper],
                'Iron': [self.iron],
                'Magnesium': [self.mag],
                'Manganese': [self.magna],
                'Phosphorus': [self.ph],
                'Potassium': [self.po],
                'Selenium': [self.sel],
                'Zinc': [self.zinc],
                'Caloric Value': [self.cv]
            }
            
            return pd.DataFrame(custom_data_input_dict)
        except Exception as e:
            raise CustomException(e,sys)