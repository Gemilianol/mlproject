import os
import sys
from src.exception import CustomException
from src.logger import logging
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass

from src.components.data_transformation import DataTransformation, DataTransformationConfig

@dataclass
class DataIngestionConfig:
    raw_data_path: str=os.path.join('artifacts','raw_data.csv')
    train_data_path: str=os.path.join('artifacts','train_data.csv')
    test_data_path: str=os.path.join('artifacts','test_data.csv')
    
class DataIngestion:
    def __init__(self):
        self.ingestion_config=DataIngestionConfig()
        
    def initiate_data_ingestion(self):
        logging.info("Entered the data ingestion method or component")
        
        try:
            df = pd.read_csv('notebook/data/raw_data.csv')
            logging.info("Read the dataset as DataFrame")
            
            os.makedirs(os.path.dirname(self.ingestion_config.raw_data_path), exist_ok=True)
            
            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)
            
            logging.info('Train Test Split initiated')
            
            train,test = train_test_split(df, test_size=0.2, random_state=42)
            train.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            test.to_csv(self.ingestion_config.test_data_path, index=False, header=True)
            
            logging.info('Ingestion of the data s completed')
            
            return(
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
        )
            
        except Exception as e:
            raise CustomException(e,sys)
        

if __name__ == "__main__":
    di = DataIngestion()
    train_data, test_data = di.initiate_data_ingestion()
    
    data_transformation = DataTransformation()
    data_transformation.initiate_data_transformation(train_data, test_data)