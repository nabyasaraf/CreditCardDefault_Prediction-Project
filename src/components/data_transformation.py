import sys
from dataclasses import dataclass
from src.components.data_ingestion import DataIngestion

import numpy as np 
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from src.exception import CustomException
from src.logger import logging
import os
from src.utils import save_object


@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path=os.path.join('artifacts','preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()


    def get_data_transformation_object(self):
        try:
            logging.info('Data Transformation initiated')
            # Define which columns should be ordinal-encoded and which should be scaled
            categorical_cols = ['SEX', 'EDUCATION', 'MARRIAGE', 'PAY_0', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6']
            numerical_cols = ['LIMIT_BAL','AGE' , 'BILL_AMT1', 'BILL_AMT2', 'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6', 'PAY_AMT1', 'PAY_AMT2', 'PAY_AMT3','PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6']
            
            
            logging.info('Pipeline Initiated')

            ## Numerical Pipeline
            num_pipeline=Pipeline(
                steps=[
                ('imputer',SimpleImputer(strategy='median'))

                ]

            )

            # Categorical Pipeline
            cat_pipeline=Pipeline(
                steps=[
                ('imputer',SimpleImputer(strategy='most_frequent'))
                
                ]

            )

            preprocessor=ColumnTransformer([
            ('num_pipeline',num_pipeline,numerical_cols),
            ('cat_pipeline',cat_pipeline,categorical_cols)
            ])
            
            return preprocessor

            logging.info('Pipeline Completed')

        except Exception as e:
            logging.info("Error in Data Transformation")
            raise CustomException(e,sys)
        
     
    def initiate_data_transformation(self,train_path,test_path):
        try:
            # Reading train and test data
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info('Read train and test data completed')
            logging.info(f'Train Dataframe Head : \n{train_df.head().to_string()}')
            logging.info(f'Test Dataframe Head  : \n{test_df.head().to_string()}')

            logging.info('Obtaining preprocessing object')

            preprocessing_obj = self.get_data_transformation_object()

            target_column_name = 'default payment next month'
            drop_columns = [target_column_name]

            
            input_feature_train_df = train_df.drop(columns=drop_columns,axis=1)
            target_feature_train_df=train_df[target_column_name]

            input_feature_test_df=test_df.drop(columns=drop_columns,axis=1)
            target_feature_test_df=test_df[target_column_name]


            #Changing the 0,5 and 6 in values to Others which is 4 as these numbers are outside the range of usual
            input_feature_train_df['EDUCATION'] = input_feature_train_df['EDUCATION'].replace({0:4,5:4,6:4,4:4})
            input_feature_test_df['EDUCATION'] = input_feature_test_df['EDUCATION'].replace({0:4,5:4,6:4,4:4})

            #Changing the 0 in values to Others which is 3 as this number is outside the range of usual
            input_feature_train_df['MARRIAGE'] = input_feature_train_df['MARRIAGE'].replace({0:3,3:3})
            input_feature_test_df['MARRIAGE'] = input_feature_test_df['MARRIAGE'].replace({0:3,3:3})

            #Tyepcasting to string
            columns_to_convert = ['SEX', 'EDUCATION', 'MARRIAGE','PAY_0','PAY_2','PAY_3','PAY_4','PAY_5','PAY_6']
            for col in columns_to_convert:
                input_feature_train_df[col] =input_feature_train_df[col].astype(str)

            for col in columns_to_convert:
                input_feature_test_df[col] =input_feature_test_df[col].astype(str)
            
            target_feature_train_df=target_feature_train_df.astype(str)
            target_feature_test_df=target_feature_test_df.astype(str)

            #Changing the 0 and -2 in values to Pay duly and payment delay for 2 months which is -1 and 2 as this number is outside the range of usual
            input_feature_train_df['PAY_0'] =  input_feature_train_df['PAY_0'].replace({0:-1,-2:2})
            input_feature_train_df['PAY_2'] =  input_feature_train_df['PAY_2'].replace({0:-1,-2:2})
            input_feature_train_df['PAY_3'] =  input_feature_train_df['PAY_3'].replace({0:-1,-2:2})
            input_feature_train_df['PAY_4'] =  input_feature_train_df['PAY_4'].replace({0:-1,-2:2})
            input_feature_train_df['PAY_5'] =  input_feature_train_df['PAY_5'].replace({0:-1,-2:2})
            input_feature_train_df['PAY_6'] =  input_feature_train_df['PAY_6'].replace({0:-1,-2:2})

            input_feature_test_df['PAY_0'] =  input_feature_test_df['PAY_0'].replace({0:-1,-2:2})
            input_feature_test_df['PAY_2'] =  input_feature_test_df['PAY_2'].replace({0:-1,-2:2})
            input_feature_test_df['PAY_3'] =  input_feature_test_df['PAY_3'].replace({0:-1,-2:2})
            input_feature_test_df['PAY_4'] =  input_feature_test_df['PAY_4'].replace({0:-1,-2:2})
            input_feature_test_df['PAY_5'] =  input_feature_test_df['PAY_5'].replace({0:-1,-2:2})
            input_feature_test_df['PAY_6'] =  input_feature_test_df['PAY_6'].replace({0:-1,-2:2})


             ## Transforming using preprocessor obj
            input_feature_train_arr=preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr=preprocessing_obj.transform(input_feature_test_df)

            logging.info("Applying preprocessing object on training and testing datasets.")

            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]
            

            save_object(

                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj

            )
            logging.info('Preprocessor pickle file saved')


            return (
               train_arr,
               test_arr,
               self.data_transformation_config.preprocessor_obj_file_path,
            )
            
        except Exception as e:
            logging.info("Exception occured in the initiate_datatransformation")

            raise CustomException(e,sys)

 