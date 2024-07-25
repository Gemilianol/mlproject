import os
import sys
import numpy as np
import pandas as pd
from src.exception import CustomException
import dill

def save_object(filepath,obj):
    try:
        dir_path = os.path.dirname(filepath)
        
        os.makedirs(dir_path, exist_ok=True)
        
        with open(filepath, "wb") as file_obj:
            dill.dump(obj, file_obj)
    
    except Exception as e:
        raise CustomException(e, sys)
    
def evaluate_models(X_train, y_train, X_test, y_test, models):
    try:
        pass
    except Exception as e:
        raise CustomException(e, sys)