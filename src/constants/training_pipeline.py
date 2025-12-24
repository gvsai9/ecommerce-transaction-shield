# constant/training_pipeline.py
import os
import sys

TARGET_COLUMN = "Is Fraudulent"

PIPELINE_NAME: str = "ecommerce_fraud_detection_pipeline"
ARTIFACT_DIR: str = "artifacts"

FILE_NAME: str = "transactions.csv"

TRAIN_FILE_NAME: str = "train.csv"
TEST_FILE_NAME: str = "test.csv"

SCHEMA_FILE_PATH = os.path.join("data_schema", "schema.yaml")

SAVED_MODEL_DIR = "saved_models"
MODEL_FILE_NAME = "model.pkl"
