# import os, sys
# from sqlalchemy import create_engine
# import pandas as pd
# import numpy as np
# from dotenv import load_dotenv

# from src.exception import CustomException, error_message_details
# from src.logging import logging
# from src.constants.config_entity import DataIngestionConfig
# from src.utils.utils import read_data_from_pg



# class DataIngestion:
#     def __init__(self):
#         data_ingestion_config = DataIngestionConfig()
#         self.table_name = data_ingestion_config.data_table_name
#         self.data_ingestion_path = data_ingestion_config.data_ingestion_config
#         self.data_ingestion_csv_path = os.path.join(self.data_ingestion_path, data_ingestion_config.data_csv_config)

#         ## create the data ingestion file if not exists
#         os.makedirs(self.data_ingestion_path, exist_ok=True)

#     def initiate_data_ingestion(self, password, username, host, port, name):
#         try:
#             logging.info('=' * 50)
#             logging.info('INITIATED DATA INGESTION')
#             logging.info('-' * 50)
#             try:
#                 logging.info('-- Starting reading the data present in the database')

#                 logging.info('---- Reading Neo Table')
#                 neo_df = read_data_from_pg(username, password, host, port, name, self.table_name)
#                 logging.info(f'---- Shape of the Neo data is {neo_df.shape}')

#                 logging.info('-- Storing the data in located directory')

#                 neo_df.to_csv(self.data_ingestion_csv_path, index=False)
#                 logging.info('---- Neo Data Stored')


#             except Exception as e:
#                 logging.error(error_message_details(e, sys))
#                 raise CustomException(e, sys)
            
#             return neo_df

#         except Exception as e:
#             logging.error(error_message_details(e, sys))
#             raise CustomException(e, sys)




import os, sys
import pandas as pd
from dotenv import load_dotenv

from src.exception import CustomException, error_message_details
from src.logging import logging
from src.constants.config_entity import DataIngestionConfig
from src.utils.utils import read_data_from_pg


class DataIngestion:
    def __init__(self):
        self.config = DataIngestionConfig()
        self.table_name = self.config.data_table_name
        self.data_ingestion_path = self.config.data_ingestion_config
        self.data_ingestion_csv_path = os.path.join(
            self.data_ingestion_path, self.config.data_csv_config
        )

        # Ensure directory exists
        os.makedirs(self.data_ingestion_path, exist_ok=True)

    def initiate_data_ingestion(self, password, username, host, port, name):
        try:
            logging.info("=" * 50)
            logging.info("INITIATED DATA INGESTION")
            logging.info("-" * 50)

            logging.info("-- Starting reading the data present in the database")
            logging.info("---- Reading Neo Table")

            neo_df = read_data_from_pg(
                username, password, host, port, name, self.table_name
            )
            logging.info(f"---- Shape of the Neo data is {neo_df.shape}")

            logging.info("-- Storing the data in located directory")
            neo_df.to_csv(self.data_ingestion_csv_path, index=False)
            logging.info(f"---- Neo Data Stored at {self.data_ingestion_csv_path}")

            return neo_df

        except Exception as e:
            logging.error(error_message_details(e, sys))
            raise CustomException(e, sys)
