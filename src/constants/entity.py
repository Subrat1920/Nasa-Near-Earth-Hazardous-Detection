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
        super().__init__()
        self.DATA_TRANSFORMATION_DIR_PATH: str = 'data_transformation'
        self.PREPROCESSING_PICKLE_TABLE: str = 'preprocessing_table'
        self.LABEL_ENCODER_PICKLE_TABLE:  str = 'label_encoder_table'
        self.REQUIRED_COLUMNS: list = ['absolute_magnitude_h', 'min_diameter_m','max_diameter_m', 'epoch_date_close_approach','miss_distance_km', 'relative_velocity_kph', 'is_potentially_hazardous','is_sentry_object']
        self.FEATURE_COLUMNS: list = ['absolute_magnitude_h', 'epoch_date_close_approach',	'miss_distance_km',	'relative_velocity_kph', 'is_sentry_object', 'diameter_range']
        self.TARGET_COLUMNS: list = ['is_potentially_hazardous']

        ## PREPROCESSIN NUMERICAL AND CATEGORICAL COLUMNS
        self.CATEGORICAL_COLUMNS: list = ['is_sentry_object']
        self.NUMERICAL_COLUMNS: list = ['absolute_magnitude_h', 'epoch_date_close_approach', 'miss_distance_km', 'relative_velocity_kph','diameter_range']
        

"""-------------------------------------MODEL TRAINING ARTIFACT ENTITY--------------------------------------"""


