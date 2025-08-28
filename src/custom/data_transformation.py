import os, sys
import pandas as pd
import numpy as np

from src.logging import logging
from src.exception import CustomeException, error_message_details
from src.constants.config_entity import DataTransformationConfig

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

class DataTransformation:
    def __init__(self):
        data_transformtaion_config = DataTransformationConfig()
        self.artifact_dir = data_transformtaion_config.artifact_path
        self.transformation_dir = data_transformtaion_config.data_transformation_path
        self.preprocessor_obj = data_transformtaion_config.preprocessor_pickle_file

        ## let's create a directory before we start
        os.makedirs(self.transformation_dir, exist_ok=True)
    
    def initiate_churn_data_transformation(churn_data):
        
