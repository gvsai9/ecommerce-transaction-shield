import os
import sys
import pandas as pd
from sklearn.model_selection import train_test_split

from src.logger import logger
from src.exception import CustomException
from src.constants import data_ingestion
from src.entity.artifact_entity import DataIngestionArtifact
from src.entity.config_entity import DataIngestionConfig


class DataIngestion:
    def __init__(self, data_ingestion_config: DataIngestionConfig):
        try:
            self.config = data_ingestion_config
        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_ingestion(self) -> DataIngestionArtifact:
        logger.info("Starting data ingestion process")

        try:
            # 1. Read raw data
            logger.info(f"Reading raw data from: {self.config.raw_data_path}")
            df = pd.read_csv(self.config.raw_data_path)

            # 2. Create artifact directory
            os.makedirs(self.config.data_ingestion_dir, exist_ok=True)

            # 3. Train-test split
            logger.info(f"Performing train-test split with test size: {data_ingestion.DATA_INGESTION_TRAIN_TEST_SPLIT_RATIO} and random state: {data_ingestion.DATA_INGESTION_RANDOM_STATE}")
            train_df, test_df = train_test_split(
                df,
                test_size=data_ingestion.DATA_INGESTION_TRAIN_TEST_SPLIT_RATIO,
                random_state=data_ingestion.DATA_INGESTION_RANDOM_STATE
            )

            # 4. Save outputs
            logger.info(f"Saving train and test datasets to artifact directory {self.config.data_ingestion_dir}")
            train_df.to_csv(self.config.train_file_path, index=False)
            test_df.to_csv(self.config.test_file_path, index=False)

            logger.info("Data ingestion completed successfully")

            return DataIngestionArtifact(
                train_file_path=self.config.train_file_path,
                test_file_path=self.config.test_file_path,
                artifact_dir=self.config.data_ingestion_dir
            )

        except Exception as e:
            logger.error("Error occurred during data ingestion")
            logger.exception(e)
            raise CustomException(e, sys)
