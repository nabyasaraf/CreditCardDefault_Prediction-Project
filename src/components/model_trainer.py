import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from src.exception import CustomException
from src.logger import logging
import pickle

from src.utils import save_object
from src.utils import evaluate_model

from dataclasses import dataclass
import sys
import os


@dataclass 
class ModelTrainerConfig:
    trained_model_file_path = os.path.join('artifacts','model.pkl')

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_training(self,train_array,test_array):
        try:
            logging.info('Splitting Dependent and Independent variables from train and test data')
            X_train, y_train, X_test, y_test = (
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )

            models={
            'DecisionTreeClassifier':DecisionTreeClassifier(),
            'RandomForestClassifier':RandomForestClassifier()
            }
            
            model_report:dict=evaluate_model(X_train,y_train,X_test,y_test,models)
            print(model_report)
            print('\n====================================================================================\n')
            logging.info(f'Model Report : {model_report}')

            best_model_score = max(model_report.values())
            selected_model=[md for md, score in model_report.items() if score == best_model_score]
            print("Best model score:", best_model_score)
            print("Best model(s):", selected_model)

            logging.info(f'Best Model Found , Model Name : {selected_model} ,Accuracy Score : {best_model_score}')

            #Extracting the model from list comprehension
            my_list = [item for item in selected_model]
            if len(my_list) == 1:
                my_list = my_list[0]
           

            save_object(
                 file_path=self.model_trainer_config.trained_model_file_path,
                 obj= my_list

            )
          

        except Exception as e:
            logging.info('Exception occured at Model Training')
            raise CustomException(e,sys)

  