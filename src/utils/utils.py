from sqlalchemy import create_engine
import requests
import pandas as pd
import logging
import sys
from src.exception import CustomException

def create_engine_for_database(user_name, password, host, port, database_name):
    engine = create_engine(
        f"postgresql+psycopg2://{user_name}:{password}@{host}:{port}/{database_name}",
        connect_args={
                "sslmode": "require",
                "keepalives": 1,
                "keepalives_idle": 30,
                "keepalives_interval": 10,
                "keepalives_count": 5
            }
    )
    return engine


def read_data_from_pg(user_name, password, host, port, database_name, table_name):
    try:
        logging.info("Creating Engine")

        engine = create_engine(
            f"postgresql+psycopg2://{user_name}:{password}@{host}:{int(port)}/{database_name}",
            connect_args={
                "sslmode": "require",
                "keepalives": 1,
                "keepalives_idle": 30,
                "keepalives_interval": 10,
                "keepalives_count": 5
            }
        )

        logging.info("Engine Created")

        query = f'SELECT * FROM "{table_name}"'
        logging.info(f"Executing query: {query}")

        df = pd.read_sql(query, engine)
        logging.info(f"Data read from the database with table name {table_name} having shape: {df.shape}")

        return df

    except Exception as e:
        logging.error(f"Error while reading the database with {table_name}: {e}")
        raise CustomException(e, sys)


def fetch_data(start_date, end_date, api):
    url = f"https://api.nasa.gov/neo/rest/v1/feed?start_date={start_date}&end_date={end_date}&api_key={api}"
    response = requests.get(url)
    if response.status_code == 200:
        return response.json()
    else:
        print('Unable to connect')