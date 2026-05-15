import os
import sys

from dataclasses import dataclass

from src.logger import logging
from src.exception import CustomException
from src.utils import evaluate_model, save_object

from catboost import CatBoostRegressor
from sklearn.ensemble import (
AdaBoostRegressor,
GradientBoostingRegressor,
RandomForestRegressor
)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

@dataclass
class ModelTrainerConfig:
    trained_model_path=os.path.join('artifacts', 'model.pkl')

class ModelTrainer:
    try:
        def __init__(self):
            self.model_training_config=ModelTrainerConfig()

        def initiate_model_trainer(self, train_array, test_array):
            X_train, y_train, X_test, y_test=(
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )

            models = {
                "Random Forest": RandomForestRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "Linear Regression": LinearRegression(),
                "XGBRegressor": XGBRegressor(),
                "CatBoosting Regressor": CatBoostRegressor(verbose=False),
                "AdaBoost Regressor": AdaBoostRegressor(),
            }

            model_report:dict=evaluate_model(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, models=models)

            best_model_score=max(sorted(model_report.values()))
            best_model_name=list(model_report.keys())[list(model_report.values()).index(best_model_score)]
            best_model=models[best_model_name]

            if best_model_score<0.6:
                raise CustomException("Model is not good")

            save_object(
                file_path=self.model_training_config.trained_model_path,
                obj=best_model
            )

            predicated=best_model.predict(X_test)
            model_r2_score=r2_score(y_test, predicated)

            return best_model_name, model_r2_score

    except Exception as e:
        raise CustomException(e, sys)
