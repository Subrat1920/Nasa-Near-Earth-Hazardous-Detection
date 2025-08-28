import os, sys
from sqlalchemy import create_engine
import pandas as pd
import numpy as np
from dotenv import load_dotenv

from src.exception import CustomeException, error_message_details
from src.logging import logging
from src.constants.config_entity import DataIngestionConfig
from src.constants.entity import TableNameEntity
from src.utils.utils import read_data_from_pg



class DataIngestion:
    def __init__(self):
        data_ingestion_config = DataIngestionConfig()
        self.data_ingestion_path = data_ingestion_config.dataset_dir_path
        self.churn_data_path = data_ingestion_config.churn_data_path
        self.customer_data_path = data_ingestion_config.customer_data_path
        self.revenue_data_path = data_ingestion_config.revenue_data_path

        table_name = TableNameEntity()
        self.churn_table = table_name.CHURN_DATA_TABLE_NAME
        self.customer_table = table_name.CUSTOMER_DATA_TABLE_NAME
        self.revenue_table = table_name.DAILY_REVENUE_TABLE_NAME

        os.makedirs(self.data_ingestion_path, exist_ok=True)

    def initiate_data_ingestion(self, password, username, host, port, name):
        try:
            logging.info('=' * 50)
            logging.info('INITIATED DATA INGESTION')
            logging.info('-' * 50)
            try:
                logging.info('-- Starting reading the data present in the database')

                logging.info('---- Reading Churn Table')
                churn_df = read_data_from_pg(username, password, host, port, name, self.churn_table)
                logging.info(f'---- Shape of the churn data is {churn_df.shape}')

                logging.info('---- Reading Customer Table')
                customer_df = read_data_from_pg(username, password, host, port, name, self.customer_table)
                logging.info(f'---- Shape of the customer data is {customer_df.shape}')

                logging.info('---- Reading Revenue Table')
                revenue_df = read_data_from_pg(username, password, host, port, name, self.revenue_table)
                logging.info(f'---- Shape of the revenue data is {revenue_df.shape}')

                logging.info('-- Storing the data in located directory')

                churn_df.to_csv(self.churn_data_path, index=False)
                logging.info('---- Churn Data Stored')

                customer_df.to_csv(self.customer_data_path, index=False)
                logging.info('---- Customer Data Stored')

                revenue_df.to_csv(self.revenue_data_path, index=False)
                logging.info('---- Revenue Data Stored')

            except Exception as e:
                logging.error(error_message_details(e, sys))
                raise CustomeException(e, sys)
            
            return churn_df, customer_df, revenue_df

        except Exception as e:
            logging.error(error_message_details(e, sys))
            raise CustomeException(e, sys)
