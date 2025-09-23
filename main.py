import os
import sys
import json
import warnings
from datetime import datetime
from dotenv import load_dotenv
from src.exception import CustomException
from src.logging import logging
from src.custom.data_ingestion import DataIngestion
from src.custom.data_transformation import DataTransformation
from src.custom.model_trainer import ModelTrainer

warnings.filterwarnings("ignore")
load_dotenv()

password = os.getenv('POSTGRES_PASSWORD')
username = os.getenv('POSTGRES_USER')
host = os.getenv('POSTGRES_HOST')
port = os.getenv('POSTGRES_PORT')
name = os.getenv('POSTGRES_DB')

# Ensure artifacts folder exists
os.makedirs("artifacts", exist_ok=True)

def write_metadata(file_path, metadata):
    with open(file_path, "w") as f:
        json.dump(metadata, f, indent=4)

if __name__ == '__main__':
    try:
        print("==== Data Ingestion Begins ====")
        data_ingestion = DataIngestion()
        neo_df = data_ingestion.initiate_data_ingestion(
            password=password,
            username=username,
            host=host,
            port=port,
            name=name
        )
        logging.info("Data ingestion completed successfully.")

        # Generate ingestion metadata
        ingestion_metadata = {
            "artifact_type": "DB table",
            "table_name": data_ingestion.table_name,
            "last_ingested": datetime.now().isoformat(),
            "rows": neo_df.shape[0],
            "columns": neo_df.shape[1],
            "description": "Weekly snapshot of NEO data pulled from NASA API via PostgreSQL"
        }
        write_metadata("artifact\data_ingestion_metadata.json", ingestion_metadata)


        print("==== Data Transformation Begins ====")
        data_transformation = DataTransformation()
        x_resample, x_test_encoded, y_resample, y_test_encoded = data_transformation.initiate_data_transformation(neo_df)
        logging.info("Data transformation completed successfully.")

        # Generate transformation metadata
        transform_metadata = {
            "artifact_type": "DB tables",
            "tables": ["preprocessor", "label_encoder"],
            "last_updated": datetime.now().isoformat(),
            "description": "Preprocessor and label encoder stored in PostgreSQL"
        }
        write_metadata("artifact\data_trasformation_metadata.json", transform_metadata)


        print("==== Model Training Begins ====")
        model_trainer = ModelTrainer()
        model_trainer.model_training_with_mlflow(
            x_train=x_resample,
            y_train=y_resample,
            x_test=x_test_encoded,
            y_test=y_test_encoded
        )
        logging.info("Model training and MLflow logging completed successfully.")

        # Generate model metadata
        model_metadata = {
            "artifact_type": "DB table",
            "table_name": "model_artifact",
            "mlflow_run_id": model_trainer.mlflow_run_id if hasattr(model_trainer, "mlflow_run_id") else None,
            "last_updated": datetime.now().isoformat(),
            "description": "Best model artifact logged in MLflow and stored in DB"
        }
        write_metadata("artifacts/model_trainer_metadata.json", model_metadata)


    except Exception as e:
        logging.error("Pipeline execution failed.")
        raise CustomException(e, sys)
