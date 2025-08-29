from sqlalchemy import create_engine
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
import requests, pickle, os
import pandas as pd
import logging
import sys
from src.exception import CustomException
from dotenv import load_dotenv
load_dotenv()

def load_pickle_from_db(table_name, db_url= os.getenv('DATABASE_URL')):
    # Create engine
    engine = create_engine(db_url)
    
    # Fetch the latest artifact
    query = f"""
    SELECT * 
    FROM {table_name}
    ORDER BY created_at DESC
    LIMIT 1
    """
    df = pd.read_sql(query, engine)

    if df.empty:
        raise ValueError(f"No artifacts found in table {table_name}")

    artifact_hex = df.iloc[0]["artifact"]

    # Convert from Postgres BYTEA hex ("\x...") to raw bytes
    artifact_bytes = bytes.fromhex(artifact_hex[2:])

    # Unpickle
    obj = pickle.loads(artifact_bytes)

    return obj

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



def get_mlflow_metrics(actual, predicted):
    accuracy = accuracy_score(actual, predicted)
    precision = precision_score(actual, predicted)
    f1 = f1_score(actual, predicted)
    return accuracy, precision, f1

