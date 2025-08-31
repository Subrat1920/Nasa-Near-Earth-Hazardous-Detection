import os
import sys
import tempfile
import json
import joblib
import numpy as np
from datetime import datetime

import mlflow
from mlflow.models import infer_signature

from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import classification_report, confusion_matrix

from xgboost import XGBClassifier
from catboost import CatBoostClassifier

import dagshub
from dotenv import load_dotenv

from src.constants.config_entity import ModelTrainerConfig
from src.exception import CustomException, error_message_details
from src.logging import logging
from src.utils.utils import get_mlflow_metrics

load_dotenv()

REPO_OWNER_NAME = os.getenv("DAGS_REPO_OWNER_NAME")
REPO_NAME = os.getenv("DAGS_REPO_NAME")
MLFLOW_REMOTE_TRACKING_URL = os.getenv("MLFLOW_REMOTE_TRACKING_URL")

# Initialize DagsHub + MLflow
dagshub.init(repo_owner=REPO_OWNER_NAME, repo_name=REPO_NAME, mlflow=True)
# Use set_tracking_uri since your code used set_tracking_uri previously
if MLFLOW_REMOTE_TRACKING_URL:
    mlflow.set_tracking_uri(MLFLOW_REMOTE_TRACKING_URL)

ml_run_name = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')


def _save_model_to_file(model, filepath):
    """
    Save model to filepath depending on model type.
    - sklearn-compatible: joblib.dump
    - CatBoostClassifier: model.save_model
    - xgboost booster: booster.save_model
    """
    # CatBoost
    try:
        if isinstance(model, CatBoostClassifier):
            model.save_model(filepath)
            return filepath
    except Exception:
        pass

    # XGBoost booster (sometimes user passes XGBClassifier or its booster)
    try:
        # If model has get_booster (XGBClassifier), save booster via save_model
        if hasattr(model, "get_booster"):
            booster = model.get_booster()
            booster.save_model(filepath)
            return filepath
    except Exception:
        pass

    # Fallback to joblib for sklearn-compatible objects
    joblib.dump(model, filepath)
    return filepath


class ModelTrainer:
    def __init__(self):
        # expects: list of tuples -> [(model_name, estimator, param_distributions), ...]
        model_train_config = ModelTrainerConfig()
        self.models_with_params = model_train_config.model_params
        self.hyp_parameter_scores = model_train_config.parameter_scoring

    def model_training_with_mlflow(self, x_train, y_train, x_test, y_test):
        """
        Trains multiple models with RandomizedSearchCV, logs all runs to MLflow,
        also trains baseline CatBoost and XGBoost, and logs the best model as artifacts
        (compatible with DagsHub's MLflow tracking).
        """
        try:
            mlflow.set_experiment(experiment_name=f"NASA NEO Training on {ml_run_name}")

            best = {
                "avg_score": -np.inf,
                "model": None,
                "name": None,
                "params": None,
                "metrics": None,
                "y_pred_test": None,
            }

            # ---- Hyperparameter-tuned models ----
            for model_name, model, param_grid in self.models_with_params:
                search = RandomizedSearchCV(
                    estimator=model,
                    param_distributions=param_grid,
                    scoring=self.hyp_parameter_scores,
                    refit='average_precision', 
                    n_jobs=-1,
                    verbose=1,
                    n_iter=10,
                    random_state=42,
                )
                search.fit(x_train, y_train)

                tuned_model = search.best_estimator_
                tuned_params = search.best_params_

                y_pred_train = tuned_model.predict(x_train)
                y_pred_test = tuned_model.predict(x_test)

                acc, prec, f1 = get_mlflow_metrics(y_test, y_pred_test)
                avg_score = (acc + prec + f1) / 3.0

                # Start MLflow run for this tuned model
                with mlflow.start_run(run_name=f"Tuned | {model_name}"):
                    # Log params & metrics
                    mlflow.log_params(tuned_params)
                    mlflow.log_metrics({"accuracy": acc, "precision": prec, "f1_score": f1})

                    # Save model to temp file and log as artifact (avoids model-registry endpoints)
                    with tempfile.TemporaryDirectory() as tmpdir:
                        model_file = os.path.join(tmpdir, f"tuned_{model_name.replace(' ', '_')}.model")
                        _save_model_to_file(tuned_model, model_file)
                        mlflow.log_artifact(local_path=model_file, artifact_path=f"models/tuned_{model_name.replace(' ', '_')}")

                        # Save input_example and signature (for reference)
                        try:
                            input_example = np.array(x_train[:5])
                            input_example_path = os.path.join(tmpdir, "input_example.npy")
                            np.save(input_example_path, input_example)
                            mlflow.log_artifact(input_example_path, artifact_path=f"models/tuned_{model_name.replace(' ', '_')}")
                        except Exception:
                            pass

                        try:
                            sig = infer_signature(x_train[:50], tuned_model.predict(x_train[:50]))
                            sig_text_path = os.path.join(tmpdir, "signature.txt")
                            with open(sig_text_path, "w") as f:
                                f.write(str(sig))
                            mlflow.log_artifact(sig_text_path, artifact_path=f"models/tuned_{model_name.replace(' ', '_')}")
                        except Exception:
                            pass

                # Update best if this is best so far
                if avg_score > best["avg_score"]:
                    best.update(
                        {
                            "avg_score": avg_score,
                            "model": tuned_model,
                            "name": model_name,
                            "params": tuned_params,
                            "metrics": {"accuracy": acc, "precision": prec, "f1_score": f1},
                            "y_pred_test": y_pred_test,
                        }
                    )

            # ---- Baseline CatBoost and XGBoost (no tuning here) ----
            baseline_models = [
                ("CatBoost Classifier", CatBoostClassifier(verbose=0, random_state=42)),
                ("XGBoost Classifier", XGBClassifier(random_state=42, n_estimators=200)),
            ]

            for model_name, model in baseline_models:
                model.fit(x_train, y_train)

                y_pred_test = model.predict(x_test)
                acc, prec, f1 = get_mlflow_metrics(y_test, y_pred_test)
                

                input_sample = x_train[:50]
                # signature computed for reference only
                try:
                    signature = infer_signature(input_sample, model.predict(input_sample))
                except Exception:
                    signature = None

                with mlflow.start_run(run_name=f"Baseline | {model_name}"):
                    # Log params if available
                    try:
                        if hasattr(model, "get_params"):
                            mlflow.log_params(model.get_params())
                    except Exception:
                        pass

                    mlflow.log_metrics({"accuracy": acc, "precision": prec, "f1_score": f1})

                    # Save baseline model to temp file and log as artifact
                    with tempfile.TemporaryDirectory() as tmpdir:
                        model_file = os.path.join(tmpdir, f"baseline_{model_name.replace(' ', '_')}.model")
                        _save_model_to_file(model, model_file)
                        mlflow.log_artifact(local_path=model_file, artifact_path=f"models/baseline_{model_name.replace(' ', '_')}")

                        # Save signature and input example as artifacts if available
                        try:
                            input_example = np.array(x_train[:5])
                            input_example_path = os.path.join(tmpdir, "input_example.npy")
                            np.save(input_example_path, input_example)
                            mlflow.log_artifact(input_example_path, artifact_path=f"models/baseline_{model_name.replace(' ', '_')}")
                        except Exception:
                            pass

                        if signature is not None:
                            try:
                                sig_text_path = os.path.join(tmpdir, "signature.txt")
                                with open(sig_text_path, "w") as f:
                                    f.write(str(signature))
                                mlflow.log_artifact(sig_text_path, artifact_path=f"models/baseline_{model_name.replace(' ', '_')}")
                            except Exception:
                                pass

                # Update best if baseline better
                if avg_score > best["avg_score"]:
                    best.update(
                        {
                            "avg_score": avg_score,
                            "model": model,
                            "name": model_name,
                            "params": getattr(model, "get_params", lambda: {})(),
                            "metrics": {"accuracy": acc, "precision": prec, "f1_score": f1},
                            "y_pred_test": y_pred_test,
                        }
                    )

            # ---- Final: log best model artifacts & metrics ----
            if best["model"] is not None:
                y_pred_best = best["y_pred_test"]
                acc, prec, f1 = get_mlflow_metrics(y_test, y_pred_best)
                cm = confusion_matrix(y_test, y_pred_best)
                cr = classification_report(y_test, y_pred_best)

                input_sample = x_train[:50]
                try:
                    signature = infer_signature(input_sample, best["model"].predict(input_sample))
                except Exception:
                    signature = None

                with mlflow.start_run(run_name=f"Best Model | {best['name']}"):
                    mlflow.log_params(best["params"] if best["params"] else {})
                    mlflow.log_metrics({"accuracy": acc, "precision": prec, "f1_score": f1})

                    # Save confusion matrix and classification report as artifacts
                    with tempfile.TemporaryDirectory() as tmpdir:
                        cm_path = os.path.join(tmpdir, "confusion_matrix.txt")
                        with open(cm_path, "w") as f:
                            f.write(str(cm))
                        mlflow.log_artifact(cm_path, artifact_path=f"models/best_{best['name'].replace(' ', '_')}")

                        cr_path = os.path.join(tmpdir, "classification_report.txt")
                        with open(cr_path, "w") as f:
                            f.write(str(cr))
                        mlflow.log_artifact(cr_path, artifact_path=f"models/best_{best['name'].replace(' ', '_')}")

                        # Save best-model file
                        model_file = os.path.join(tmpdir, f"best_{best['name'].replace(' ', '_')}.model")
                        _save_model_to_file(best["model"], model_file)
                        mlflow.log_artifact(model_file, artifact_path=f"models/best_{best['name'].replace(' ', '_')}")

                        # Save input_example
                        try:
                            input_example = np.array(x_train[:5])
                            input_example_path = os.path.join(tmpdir, "input_example.npy")
                            np.save(input_example_path, input_example)
                            mlflow.log_artifact(input_example_path, artifact_path=f"models/best_{best['name'].replace(' ', '_')}")
                        except Exception:
                            pass

                        # Save signature text if available
                        if signature is not None:
                            try:
                                sig_text_path = os.path.join(tmpdir, "signature.txt")
                                with open(sig_text_path, "w") as f:
                                    f.write(str(signature))
                                mlflow.log_artifact(sig_text_path, artifact_path=f"models/best_{best['name'].replace(' ', '_')}")
                            except Exception:
                                pass

                    logging.info(f"Best model '{best['name']}' logged in MLflow with metrics.")
            else:
                logging.error("No best model found.")

        except Exception as e:
            logging.error("Got an error while training the model with MLflow")
            # preserve your exception flow
            raise CustomException(e, sys)
