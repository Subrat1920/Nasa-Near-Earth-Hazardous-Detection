import os
import sys
import warnings
warnings.filterwarnings("ignore")
from src.exception import CustomException, error_message_details
from src.logging import logging
from src.custom.data_ingestion import DataIngestion
from dotenv import load_dotenv
load_dotenv()

password = os.getenv('POSTGRES_PASSWORD')
username = os.getenv('POSTGRES_USER')
host = os.getenv('POSTGRES_HOST')
port = os.getenv('POSTGRES_PORT')
name = os.getenv('POSTGRES_DB')


if __name__=='__main__':
    data_ingestion = DataIngestion()
    neo_df = data_ingestion.initiate_data_ingestion(
        password=password, 
        username=username, 
        host=host, 
        port=port, 
        name=name)