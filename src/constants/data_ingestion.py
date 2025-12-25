"""
Data Ingestion related constants
All constants related to reading raw data
and writing ingested artifacts live here.
"""

import os

# Root data directory
DATA_DIR: str = "data"

# Raw input data (DVC tracked)
DATA_INGESTION_RAW_DIR: str = os.path.join(DATA_DIR, "raw")
DATA_INGESTION_RAW_FILE_NAME: str = "transactions.csv"

# Artifact directories (inside pipeline artifacts)
DATA_INGESTION_DIR_NAME: str = "data_ingestion"
DATA_INGESTION_FEATURE_STORE_DIR: str = "feature_store"
DATA_INGESTION_INGESTED_DIR: str = "ingested"

# Output file names
DATA_INGESTION_TRAIN_FILE_NAME: str = "train.csv"
DATA_INGESTION_TEST_FILE_NAME: str = "test.csv"

# Split configuration
DATA_INGESTION_TRAIN_TEST_SPLIT_RATIO: float = 0.2
DATA_INGESTION_RANDOM_STATE: int = 42
