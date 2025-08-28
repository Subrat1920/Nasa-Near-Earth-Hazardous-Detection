import os, sys
from src.constants.entity import DataIngestionEntity, ArtifactEntity, DataTransformationEntity

class ArtifactConfig:
    def __init__(self):
        artifact_entity = ArtifactEntity()
        self.artifact_name = artifact_entity.ARTIFACT_DIR_NAME


class DataIngestionConfig:
    def __init__(self):
        ## Data Ingestion Entity Object
        data_ingestion_entity = DataIngestionEntity()

        ## paths for the data ingestion class
        self.dataset_dir_path = data_ingestion_entity.DATA_INGESTION_DIR_PATH
        self.churn_data_path = os.path.join(self.dataset_dir_path, data_ingestion_entity.CHURN_DATA_FILE_NAME)
        self.customer_data_path = os.path.join(self.dataset_dir_path, data_ingestion_entity.CUSOMER_DATA_FILE_NAME)
        self.revenue_data_path =  os.path.join(self.dataset_dir_path, data_ingestion_entity.REVENUE_DATA_FILE_NAME)

class DataValidationConfig:
    def __init__(self):
        pass

class DataTransformationConfig:
    def __init__(self):
        ## Data Transformation Entity Object
        data_transformtion_entity = DataTransformationEntity()

        ## paths for transformation object
        self.artifact_path = data_transformtion_entity.ARTIFACT_DIR_NAME
        self.data_transformation_path = os.path.join(self.artifact_path, data_transformtion_entity.DATA_TRANSFORMATION_DIR_PATH)
        self.preprocessor_pickle_file = os.path.join(self.data_transformation_path, data_transformtion_entity.PREPROCESSING_PICKLE_FILE)



class ModelTrainerConfig:
    def __init__(self):
        pass


if __name__=='__main__':
    data_ingestion_config = DataIngestionConfig()
    print(data_ingestion_config.dataset_dir_path)
    print(data_ingestion_config.churn_data_path)
    print(data_ingestion_config.customer_data_path)
    print(data_ingestion_config.revenue_data_path)