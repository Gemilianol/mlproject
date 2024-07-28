import sys
import os
from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path= os.path.join('artifacts','preprocessor.pkl')
    
class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()
        
    def get_data_transformer_obj(self):
        
        """
        This function is responsable for data transformation

        Raises:
            CustomException: Returns the error message details

        Returns:
            _type_: Train and Test tranformed and Preprocessing.pkl
        """
        
        try:
            
            numerical_features = ['Fat', 'Saturated Fats',
       'Monounsaturated Fats', 'Polyunsaturated Fats', 'Carbohydrates',
       'Sugars', 'Protein', 'Dietary Fiber', 'Cholesterol', 'Sodium', 'Water',
       'Vitamin A', 'Vitamin B1', 'Vitamin B11', 'Vitamin B12', 'Vitamin B2',
       'Vitamin B3', 'Vitamin B5', 'Vitamin B6', 'Vitamin C', 'Vitamin D',
       'Vitamin E', 'Vitamin K', 'Calcium', 'Copper', 'Iron', 'Magnesium',
       'Manganese', 'Phosphorus', 'Potassium', 'Selenium', 'Zinc',
       'Caloric Value']
            # categorical_features=[]
            
            num_pipeline= Pipeline(
                steps=[
                    ('imputer',SimpleImputer(strategy='median')),
                    ('scaler', StandardScaler())
                ]
            )
            
            logging.info('Numerical columns standard scaling is completed')
            
            # cat_pipeline = Pipeline(
            #     steps=[
            #         ('imputer',SimpleImputer(strategy='most_frequent')),
            #         ('encoder',OneHotEncoder()),
            #         ('scaler',StandardScaler())
            #     ]
            # )
            
            # logging.info('Categorical columns encoding completed')
            
            preprocessor=ColumnTransformer(
                [
                    ("num_pipeline",num_pipeline,numerical_features)
                ]
            )
            
            # preprocessor=ColumnTransformer(
            #     [
            #         ("num_pipeline",num_pipeline,numerical_features),
            #         ("cat_pipeline",cat_pipeline,categorical_features)
            #     ]
            # )
            
            return preprocessor
        
        except Exception as e:
            raise CustomException(e,sys)
        
    
    def initiate_data_transformation(self, train_path,test_path):
        
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            
            logging.info("Read train and test data completed")
            
            logging.info("Obtaining Pre-Processing object")
            
            preprocessing_obj = self.get_data_transformer_obj()
            
            target_column_name = 'Nutrition Density'
            
            #I need to add a function in utils in order to drop columns data.drop(columns=['Unnamed: 0.1', 'Unnamed: 0'], inplace=True)
            
            #columns_drop = ['Unnamed: 0.1', 'Unnamed: 0','food','Caloric Value']
            
            
            numerical_features = ['Fat', 'Saturated Fats','Monounsaturated Fats', 
                                  'Polyunsaturated Fats', 'Carbohydrates',
                                  'Sugars', 'Protein', 'Dietary Fiber', 'Cholesterol', 
                                  'Sodium', 'Water','Vitamin A', 'Vitamin B1', 'Vitamin B11', 
                                  'Vitamin B12', 'Vitamin B2','Vitamin B3', 'Vitamin B5', 
                                  'Vitamin B6', 'Vitamin C', 'Vitamin D','Vitamin E', 'Vitamin K', 
                                  'Calcium', 'Copper', 'Iron', 'Magnesium','Manganese', 'Phosphorus', 
                                  'Potassium', 'Selenium', 'Zinc','Caloric Value']
            
            imput_features_train_df=train_df.drop(columns=[target_column_name], axis=1)
            target_features_train_df = train_df[target_column_name]
            
            imput_features_test_df=test_df.drop(columns=[target_column_name], axis=1)
            target_features_test_df = test_df[target_column_name]
            
            logging.info("Applying preprocessing object on train and test dataframe")
            
            imput_features_train_arr = preprocessing_obj.fit_transform(imput_features_train_df)
            imput_features_test_arr = preprocessing_obj.transform(imput_features_test_df)
            
            train_arr = np.c_[imput_features_train_arr, np.array(target_features_train_df)]
            test_arr = np.c_[imput_features_test_arr, np.array(target_features_test_df)]
            
            logging.info('Saved preprocessing object')
            
            save_object(filepath = self.data_transformation_config.preprocessor_obj_file_path,
                           obj=preprocessing_obj)
            
            return(
                train_arr, test_arr, self.data_transformation_config.preprocessor_obj_file_path
            )
        except Exception as e:
            raise CustomException(e, sys)
        
if __name__ == "__main__":
    pass