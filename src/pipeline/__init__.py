# use of model recovery
import os
import mlflow
import pandas as pd
from sqlalchemy import create_engine
from dotenv import load_dotenv
load_dotenv
DATABASE_URL = os.getenv("DATABASE_URL")

input_data = None
engine = create_engine(DATABASE_URL)
df = pd.read_sql("SELECT * FROM model_training_logs ORDER BY training_date DESC LIMIT 1", engine)

artifact_uri = df["artifact_uri"].iloc[0] + "/models/" + df["model_name"].iloc[0]
best_model = mlflow.pyfunc.load_model(artifact_uri)

preds = best_model.predict(input_data)
