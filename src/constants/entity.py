""" ------------------------------------DEFINING ALL THE TABLE NAMES USED --------------------------------"""

class TableNameEntity:
    def __init__(self):        
        self.DAILY_REVENUE_TABLE_NAME: str = 'daily_revenue'

""" ------------------------------------ARTIFACT DIRECTORY -------------------------------------------------"""

class ArtifactEntity:
    def __init__(self):
        self.ARTIFACT_DIR_NAME : str = 'artifacts'

""" ------------------------------------DATA INGESTION ARTIFACT ---------------------------------------------"""
class DataIngestionEntity:
    def __init__(self):
        self.DATA_INGESTION_DIR_PATH : str = 'Datasets'
        self.CHURN_DATA_FILE_NAME : str = 'churn_data.csv'
        self.CUSOMER_DATA_FILE_NAME: str = 'customer_data.csv'
        self.REVENUE_DATA_FILE_NAME : str = 'revenue_data.csv'

"""-------------------------------------DATA VALIDATION ARTIFACT ENTITY--------------------------------------"""


"""-------------------------------------DATA TRANSFORMATION ARTIFACT ENTITY--------------------------------------"""
class DataTransformationEntity(ArtifactEntity):
    def __init__(self):
        self.DATA_TRANSFORMATION_DIR_PATH: str = 'data_transformation'
        self.PREPROCESSING_PICKLE_FILE: str = 'churn_preprocessor.pkl'

"""-------------------------------------MODEL TRAINING ARTIFACT ENTITY--------------------------------------"""


