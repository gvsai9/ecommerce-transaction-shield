import os
import sys
import pandas as pd
from scipy.stats import ks_2samp

from src.exception import CustomException
from src.logger import logging
from src.constants import training_pipeline
from src.entity.artifact_entity import (
    DataIngestionArtifact,
    DataValidationArtifact,
)
from src.entity.config_entity import DataValidationConfig
from src.utils import read_yaml_file, write_yaml_file


class DataValidation:
    def __init__(
        self,
        data_ingestion_artifact: DataIngestionArtifact,
        data_validation_config: DataValidationConfig,
    ):
        try:
            self.data_ingestion_artifact = data_ingestion_artifact
            self.data_validation_config = data_validation_config
            self.schema_config = read_yaml_file(training_pipeline.SCHEMA_FILE_PATH)
        except Exception as e:
            logging.error(f"Error in DataValidation init: {e}")
            raise CustomException(e, sys)

    @staticmethod
    def read_data(file_path: str) -> pd.DataFrame:
        try:
            return pd.read_csv(file_path)
        except Exception as e:
            raise CustomException(e, sys)

    def validate_schema(self, dataframe: pd.DataFrame) -> bool:
        schema_columns = set(self.schema_config["columns"].keys())
        dataframe_columns = set(dataframe.columns)
        logging.info("Validating schema...")
        logging.info(f"Schema columns: {schema_columns}")
        logging.info(f"Dataframe columns: {dataframe_columns}")

        return schema_columns == dataframe_columns

    def detect_dataset_drift(
        self, base_df: pd.DataFrame, current_df: pd.DataFrame, threshold=0.03
    ) -> bool:
        status = True
        report = {}
        logging.info("Detecting dataset drift...")
        for column in base_df.columns:
            d1 = base_df[column]
            d2 = current_df[column]

            ks_result = ks_2samp(d1, d2)

            drift_detected = bool(ks_result.pvalue < threshold)
            if drift_detected:
                status = False
            report[column] = {
                "p_value": float(ks_result.pvalue),
                "drift_detected": drift_detected
            }


        os.makedirs(os.path.dirname(self.data_validation_config.drift_report_file_path), exist_ok=True)
        write_yaml_file(
            file_path=self.data_validation_config.drift_report_file_path,
            content=report,
        )

        return status

    def initiate_data_validation(self) -> DataValidationArtifact:
        try:
            train_path = self.data_ingestion_artifact.train_file_path
            test_path = self.data_ingestion_artifact.test_file_path

            train_df = self.read_data(train_path)
            test_df = self.read_data(test_path)

            schema_valid = self.validate_schema(train_df) and self.validate_schema(test_df)

            drift_status = self.detect_dataset_drift(train_df, test_df)

            validation_status = schema_valid and drift_status

            if validation_status:
                os.makedirs(os.path.dirname(self.data_validation_config.valid_train_file_path), exist_ok=True)
                train_df.to_csv(self.data_validation_config.valid_train_file_path, index=False)
                test_df.to_csv(self.data_validation_config.valid_test_file_path, index=False)

                invalid_train = None
                invalid_test = None
            else:
                os.makedirs(os.path.dirname(self.data_validation_config.invalid_train_file_path), exist_ok=True)
                train_df.to_csv(self.data_validation_config.invalid_train_file_path, index=False)
                test_df.to_csv(self.data_validation_config.invalid_test_file_path, index=False)

                invalid_train = self.data_validation_config.invalid_train_file_path
                invalid_test = self.data_validation_config.invalid_test_file_path

            return DataValidationArtifact(
                validation_status=validation_status,
                valid_train_file_path=self.data_validation_config.valid_train_file_path
                if validation_status
                else None,
                valid_test_file_path=self.data_validation_config.valid_test_file_path
                if validation_status
                else None,
                invalid_train_file_path=invalid_train,
                invalid_test_file_path=invalid_test,
                drift_report_file_path=self.data_validation_config.drift_report_file_path,
            )

        except Exception as e:
            raise CustomException(e, sys)
