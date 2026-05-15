import pandas as pd
import os
import sys

from src.exception import CustomException
from src.logger import logging 

from src.components.data_transformation import DataTransformationConfig
from src.components.data_transformation import DataTransformation

from src.components.model_trainer import ModelTrainerConfig
from src.components.model_trainer import ModelTrainer

from src.components.hyperparameter_tunning import HyperparaConfig
from src.components.hyperparameter_tunning import Hyperpara

from sklearn.model_selection import train_test_split

from dataclasses import dataclass

@dataclass
class DataIngestionConfig:
    train_data_path:str=os.path.join('artifacts', 'train.csv')
    test_data_path:str=os.path.join('artifacts', 'test.csv')
    raw_data_path:str=os.path.join('artifacts', 'data.csv')

class DataIngestion:
    def __init__(self):
        self.ingestion_config=DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Data Ingestion is Started")

        df = pd.read_csv('notebook\data\stud.csv')
        logging.info("Read the dataset")

        os.makedirs(os.path.dirname(self.ingestion_config.raw_data_path), exist_ok=True)
        logging.info("Create the artifacts directory")

        train_set, test_set=train_test_split(df, test_size=0.2, random_state=42)
        logging.info("Train and Test dataset splited")

        df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)
        train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
        test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)
        logging.info("seperate the train, test and raw data in artifacts folder")

        return(
            self.ingestion_config.train_data_path,
            self.ingestion_config.test_data_path
        )
    
if __name__=='__main__':
    try:
        data_ingestion=DataIngestion()
        train_path, test_path = data_ingestion.initiate_data_ingestion()

        data_transformation=DataTransformation()
        train_arr,test_arr,_=data_transformation.initiate_data_transformation(train_path, test_path)

        model_trainer=ModelTrainer()
        best_model, model_score=model_trainer.initiate_model_trainer(train_array=train_arr, test_array=test_arr)
        print(f"Best Model: {best_model}")
        print(f"Model Accuracy: {model_score}")

        Hyper_tunning=Hyperpara()
        best_hyper_tunned_model, hyper_tunned_model_score=Hyper_tunning.initiate_hyper_tunning(train_array=train_arr, test_array=test_arr)
        print(f"Hyper Tunned Best Model: {best_hyper_tunned_model}")
        print(f"Model Accuracy: {hyper_tunned_model_score}")

    except Exception as e:
        raise CustomException(e, sys)
        

