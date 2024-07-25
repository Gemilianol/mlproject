import os
import sys

from dataclasses import dataclass

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, evaluate_model

from catboost import CatBoostRegressor

from sklearn.ensemble import(AdaBoostRegressor,GradientBoostingRegressor,RandomForestRegressor)

from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score #Double check if it's necessary this or another.
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join('artifacts','model.pkl')
    
class ModelTrainer:
    def __init__(self):
        self.model_trainer_config= ModelTrainerConfig()
        
    def initiate_model_trainer(self,train_arr,test_arr,preprocessor_path):
        try:
            logging.info("Spliting training and test input data")
            X_train, y_train, X_test, y_test = (
                train_arr[:,:-1], train_arr[:,-1],
                test_arr[:,:-1], test_arr[:,-1],
            )
            
            models = {
                "Random Forest": RandomForestRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Gradient Bossting": GradientBoostingRegressor(),
                "Linear Regression": LinearRegression(),
                "XGBoost Regressor": XGBRegressor(),
                "CatBoost Regressor": CatBoostRegressor(verbose=False),
                "AdaBoost Regressor": AdaBoostRegressor()
            }
            
            model_report: dict=evaluate_model(X_train=X_train, y_train=y_train,
                                              X_test=X_test, y_test=y_test, models=models)
            
            
            
            
        except Exception as e:
            raise CustomException(e, sys)