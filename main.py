import os
import sys
import warnings
warnings.filterwarnings("ignore")

from src.components.data_ingestion import DataIngestion
from src.components.data_validation import DataValidation
from src.components.data_transformation import DataTransformation
from src.components.model_training import ModelTrainer
# from src.components. import *

from src.entity.config_entity import (
    DataIngestionConfig,
    DataValidationConfig,
    ModelTrainerConfig,
    TrainingPipelineConfig,
    DataTransformationConfig,
)

from src.exception.exception import CustomException, error_message_details
from src.logging.logger import logging




if __name__ == "__main__":
    try:
        ## Data Ingestion
        print('Artifacts......')
        print('     -Data Ingestion......')
        trainingpiplineconfig = TrainingPipelineConfig()
        dataingestionconfig = DataIngestionConfig(trainingpiplineconfig)
        data_ingestion = DataIngestion(dataingestionconfig)
        logging.info("Initiate data ingestion")
        dataingestionartifact = data_ingestion.initiate_data_ingestion()
        logging.info("Data Ingestion Completed")

        ## Data Validation
        print('     -Data Validation......')
        data_validation_config = DataValidationConfig(trainingpiplineconfig)
        data_validation = DataValidation(dataingestionartifact, data_validation_config)
        logging.info("Initiated data validation")
        data_validation_artifact = data_validation.initiate_data_validation()
        logging.info("Data Validation Completed")

        ## Data Transformation
        print('     -Data Transformation......')
        data_transformation_config = DataTransformationConfig(trainingpiplineconfig)
        logging.info("Data Transformation started")
        data_transformation = DataTransformation(data_validation_artifact, data_transformation_config)
        data_transformation_artifact = data_transformation.initiate_data_transformation()
        logging.info("Data Transformation completed")

        ## Model Trainer
        logging.info("Model Trainer Started")
        print('     -Model Trainer......')
        model_trainer_config = ModelTrainerConfig(trainingpiplineconfig)
        model_trainer = ModelTrainer(model_trainer_config=model_trainer_config, data_transformation_artifact=data_transformation_artifact)
        model_trainer_artifact = model_trainer.initiate_model_trainer()
        logging.info("Model Trainer Completed")

        # ## run with mlflow
        # print('     -MLFLOW Runnings......')
        # mlflow_run = MlflowTrainer(model_trainer_config=model_trainer_artifact, data_transformation_artifact=data_transformation_artifact)
        # mlflow_run.initiate_model_trainer()


    except Exception as e:
        logging.error(error_message_details(e, sys))
        raise CustomException(e, sys)
