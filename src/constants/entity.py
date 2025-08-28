""" ------------------------------------DEFINING ALL THE TABLE NAMES USED --------------------------------"""

class TableNameEntity:
    def __init__(self):        
        self.NEAR_EARTH_OBJECT_TABLE: str = 'train_neo'

""" ------------------------------------ARTIFACT DIRECTORY -------------------------------------------------"""

class ArtifactEntity:
    def __init__(self):
        self.ARTIFACT_DIR_NAME : str = 'artifacts'

""" ------------------------------------DATA INGESTION ARTIFACT ---------------------------------------------"""
class DataIngestionEntity(TableNameEntity):
    def __init__(self):
        super().__init__()
        self.DATA_INGESTION_DIR_PATH : str = 'Data'
        self.NEO_DATA_FILE_NAME : str = 'neo_data.csv'


"""-------------------------------------DATA VALIDATION ARTIFACT ENTITY--------------------------------------"""


"""-------------------------------------DATA TRANSFORMATION ARTIFACT ENTITY--------------------------------------"""
class DataTransformationEntity(ArtifactEntity):
    def __init__(self):
        self.DATA_TRANSFORMATION_DIR_PATH: str = 'data_transformation'
        self.PREPROCESSING_PICKLE_FILE: str = 'churn_preprocessor.pkl'

"""-------------------------------------MODEL TRAINING ARTIFACT ENTITY--------------------------------------"""


