import requests
import os, sys
from sqlalchemy import create_engine
import pandas as pd
import logging

def fetch_data(start_date, end_date, api):
    url = f"https://api.nasa.gov/neo/rest/v1/feed?start_date={start_date}&end_date={end_date}&api_key={api}"
    response = requests.get(url)
    if response.status_code == 200:
        return response.json()
    else:
        print('Unable to connect')

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
            f"postgresql+psycopg2://{user_name}:{password}@{host}:{port}/{database_name}",
            connect_args={
                "sslmode": "require",
                "keepalives": 1,
                "keepalives_idle": 30,
                "keepalives_interval": 10,
                "keepalives_count": 5
            }
        )

        logging.info("Engine Created")

        query = f"SELECT * FROM {table_name}"
        logging.info(f"Executing query: {query}")

        df = pd.read_sql(query, engine)
        logging.info(f"Data read from the database with table name {table_name} having shape: {df.shape}")

        return df
    except Exception as e:
        logging.info("Failed to read the data from postgres")

def fetch_data_from_pg(user_name, password, host, database_name, port=5432, table_name = 'neo_table'):
    try:
        engine = create_engine(
            f"postgresql_psycopg2://{user_name}:{password}@{host}:{port}/{database_name}",
            connect_args= {
                "sslmode": "require",
                "keepalives": 1,
                "keepalives_idle": 30,
                "keepalives_interval": 10,
                "keepalives_count": 5
            }
        )
        query = f"select * from {table_name}"
        con = engine.raw_connection()
        try:
            df = pd.read_sql(query, con)
            logging.info(f'Data read with shape {df.shape}')
        except Exception as e:
            logging.error('Cannot fetch the data')

    except Exception as e:
        logging.error(f'Error {e}')
        return None