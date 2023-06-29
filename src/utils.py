import os
import sys
import pickle
import numpy as np 
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier

from src.exception import CustomException
from src.logger import logging

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)
    
def evaluate_model(X_train,y_train,X_test,y_test,models):
    try:
        report = {}
        model=DecisionTreeClassifier()

         # Train model
        model.fit(X_train,y_train)

        # Predict Testing data
        y_pred =model.predict(X_test)

        # Get accuracy scores for train and test data
        first_accuracy=accuracy_score(y_test,y_pred)
        
        #Hyper-parameter Tuning:GridSearchCV
        param_grid = {
        'max_depth': [3, 4, 5, 6],
        'min_samples_split': [2, 3, 4, 5],
        'min_samples_leaf': [1, 2, 3]
        }
        grid_search=GridSearchCV(estimator=model,param_grid=param_grid,cv=5)
        grid_search.fit(X_train,y_train)
        print(grid_search.best_estimator_)
        print(grid_search.best_params_)

        #After applying the best parameters from GridSearchCV
        model_1=DecisionTreeClassifier()
        model_1.max_depth = 6
        model_1.min_samples_leaf = 2
        model_1.min_samples_split = 4

        model_1.fit(X_train,y_train)
        y_pred1=model_1.predict(X_test)
        second_accuracy=accuracy_score(y_test,y_pred1)

        print("Model:Decison Tree")
        print(f"Accuracy:{second_accuracy}")

        #Classification Report
        print(f"GridSearchCVClassification Report : \n {classification_report(y_test,y_pred1)}")

        #Hyper-parameter Tuning:RandomizedSearchCV for RandomForestClassifier

        params={
        
        "n_estimators":[50,100,200],
        "criterion":["gini","entropy"],
        "max_depth":[3,5,10]

        }

        model_2=RandomForestClassifier()
        model_2.set_params(**{'oob_score': True})
        cv=RandomizedSearchCV(model_2,param_distributions=params,scoring='accuracy',cv=5,verbose=3)
        cv.fit(X_train,y_train)
        print(cv.best_params_)
        best_model = RandomForestClassifier()
        best_model.n_estimators = 100
        best_model.max_depth = 10
        best_model.criterion = 'gini'
        best_model.oob_score = True
        best_model.fit(X_train,y_train)
        y_pred2=best_model.predict(X_test)
        best_accuracy=accuracy_score(y_test,y_pred2)*100
        #Classification Report
        print(f"r=RandomSearchCVClassification Report : \n {classification_report(y_test,y_pred2)}")

        accuracy_report={model_1:[second_accuracy],best_model:[best_accuracy]}
        

        return accuracy_report


    except Exception as e:
        logging.info('Exception occured during model training')
        raise CustomException(e,sys)
    
def load_object(file_path):
    try:
        with open(file_path,'rb') as file_obj:
            return pickle.load(file_obj)
    except Exception as e:
        logging.info('Exception Occured in load_object function utils')
        raise CustomException(e,sys)