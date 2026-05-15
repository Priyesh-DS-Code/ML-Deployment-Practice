import sys
import os

from src.logger import logging
from src.exception import CustomException
from src.utils import save_object

import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

from dataclasses import dataclass

@dataclass
class DataTransformationConfig:
    preprocessor_file_path=os.path.join('artifacts', 'preprocessor.pkl')

class DataTransformation:
    try:
        def __init__(self):
            self.data_transformation_config=DataTransformationConfig()
    
        def get_data_transformed(self):
            num_col=['reading_score', 'writing_score']
            cat_col=[
                'gender',
                'race_ethnicity',
                'parental_level_of_education',
                'lunch',
                'test_preparation_course'
            ]
            logging.info("numerical and categorical columns are defined")

            num_pipeline=Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler())
            ])

            cat_pipeline=Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('encode', OneHotEncoder()),
                ('scaler', StandardScaler(with_mean=False))
            ])

            logging.info("Numerical and Categorical Pipeline is ready")

            preprocessor=ColumnTransformer(transformers=[
                ('num', num_pipeline, num_col),
                ('cat', cat_pipeline, cat_col)
            ])
            logging.info("Preprocessor obj is Ready")

            return preprocessor
        

        def initiate_data_transformation(self, train_set, test_set):
            train_df=pd.read_csv(train_set)
            test_df=pd.read_csv(test_set)
            logging.info("Read the train and test dataframe")

            preprocessor_obj=self.get_data_transformed()

            output_column='math_score'

            input_train_df=train_df.drop(columns=[output_column])
            output_train_df=train_df[output_column]

            input_test_df=test_df.drop(columns=[output_column])
            output_test_df=test_df[output_column]

            transformed_input_train=preprocessor_obj.fit_transform(input_train_df)
            transformed_input_test=preprocessor_obj.transform(input_test_df)

            train_arr=np.c_[transformed_input_train, np.array(output_train_df)]
            test_arr=np.c_[transformed_input_test, np.array(output_test_df)]

            save_object(
                file_path=self.data_transformation_config.preprocessor_file_path,
                obj=preprocessor_obj
            )
            logging.info("Preprocessor obj is Saved")

            return(
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_file_path
            )

    except Exception as e:
        raise CustomException(e, sys)
       




