import os, sys
import pandas as pd
import pickle

from src.logging import logging
from src.utils.utils import create_engine_for_database
from src.exception import CustomException, error_message_details
from src.constants.config_entity import DataTransformationConfig

from imblearn.over_sampling import SMOTE
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from dotenv import load_dotenv
load_dotenv()

password = os.getenv('POSTGRES_PASSWORD')
username = os.getenv('POSTGRES_USER')
host = os.getenv('POSTGRES_HOST')
port = os.getenv('POSTGRES_PORT')
name = os.getenv('POSTGRES_DB')

class DataTransformation:
    def __init__(self):
        data_transformtaion_config = DataTransformationConfig()
        self.preprocessor_table_name = data_transformtaion_config.preprocessing_table_name
        self.label_encoder_table_name = data_transformtaion_config.label_encoder_table_name
        self.features = data_transformtaion_config.features
        self.target = data_transformtaion_config.target
        self.required_features = data_transformtaion_config.required_features
        self.num_cols = data_transformtaion_config.num_cols
        self.cat_cols = data_transformtaion_config.cat_cols


    
    def gathering_required_data(self, neo_data): 
        try:
            logging.info(f'Before Required Gathering Columns: {neo_data.columns.tolist()}')
            df = neo_data[self.required_features]
            df['diameter_range'] = df['max_diameter_m'] - df['min_diameter_m']
            df = df.drop(columns=['max_diameter_m', 'min_diameter_m'])
            logging.info(f'After Required Gathering Columns: {df.columns.tolist()}')
            return df

        except Exception as e:
            raise CustomException(error_message_details(e, sys))
    
    def training_and_testing_split(self, req_data):
        try:
            x = req_data[self.features]
            y = req_data[self.target]
            logging.info(f'Independent Features: {self.features} and dependent features {self.target}')
            x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=42, test_size=0.3)
            logging.info(f'Checking the shapes x_train: {x_train.shape}, x_test: {x_test.shape}, y_train: {y_train.shape}, y_test: {y_test.shape}')
            return x_train, x_test, y_train, y_test
        except Exception as e:
            raise CustomException(error_message_details(e, sys))
        
    


    def preprocess_data(self, x_train, y_train, x_test, y_test):
        try:
            preprocessor = ColumnTransformer(
                transformers=[
                    ('num', StandardScaler(), self.num_cols),
                    ('cat', OneHotEncoder(), self.cat_cols)
                ]
            )

            ## preprocessing x_train
            x_train_encoded = preprocessor.fit_transform(x_train)
            x_test_encoded = preprocessor.transform(x_test)

            le = LabelEncoder()
            ## encoding y_train
            y_train_encoded = le.fit_transform(y_train)
            y_test_encoded = le.transform(y_test)

            ## creating an engine for storing the data in the database
            engine = create_engine_for_database(
                user_name=username, password=password, host=host, port=port, database_name=name
            )

            # serialize objects to bytes
            preprocessor_bytes = pickle.dumps(preprocessor)
            le_bytes = pickle.dumps(le)

            ## dataframe for preprocessor
            preprocessor_df = pd.DataFrame([{
                "name": "preprocessor",
                "artifact": preprocessor_bytes,
                "created_at": pd.Timestamp.now()
            }])

            ## dataframe for label encoder
            label_encoder_df = pd.DataFrame([{
                "name": "label_encoder",
                "artifact": le_bytes,
                "created_at": pd.Timestamp.now()
            }])

            ## save artifacts in database (append, keeps history)
            preprocessor_df.to_sql(
                self.preprocessor_table_name, engine, if_exists='replace', index=False
            )
            label_encoder_df.to_sql(
                self.label_encoder_table_name, engine, if_exists='replace', index=False
            )

            return x_train_encoded, y_train_encoded, x_test_encoded, y_test_encoded

        except Exception as e:
            raise CustomException(error_message_details(e, sys))


    
    def treating_imbalanced_data(self, x_train_encoded, y_train_encoded):
        try:
            smote = SMOTE()
            x_resample, y_resample = smote.fit_resample(x_train_encoded, y_train_encoded)
            return x_resample, y_resample
        except Exception as e:
            raise CustomException(error_message_details(e, sys))
    
    def initiate_data_transformation(self, data):
        try:
            logging.info("=" * 50)
            logging.info("INITIATED DATA TRANSFORMATION")
            logging.info("-" * 50)

            logging.info("-- Starting reading the data present in the database")
            logging.info("---- Reading Neo Table")

            df = self.gathering_required_data(data)
            x_train, x_test, y_train, y_test = self.training_and_testing_split(df)
            x_train_encoded, y_train_encoded, x_test_encoded, y_test_encoded = self.preprocess_data(x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test)
            x_resample, y_resample = self.treating_imbalanced_data(x_train_encoded, y_train_encoded)
            return x_resample, x_test_encoded, y_resample, y_test_encoded
        except Exception as e:
            raise CustomException(error_message_details(e, sys))