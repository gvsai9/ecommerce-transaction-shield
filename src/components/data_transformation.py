import os
import sys
import numpy as np
import pandas as pd
import pickle
from src.logger import logger
from src.exception import CustomException
from src.constants.training_pipeline import TARGET_COLUMN
from src.entity.artifact_entity import DataValidationArtifact, DataTransformationArtifact
from src.entity.config_entity import DataTransformationConfig


class DataTransformation:
    def __init__(
        self,
        data_validation_artifact: DataValidationArtifact,
        data_transformation_config: DataTransformationConfig,
    ):
        self.validation_artifact = data_validation_artifact
        self.config = data_transformation_config

    @staticmethod
    def log_transform(series: pd.Series) -> pd.Series:
        return np.log1p(series)

    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        logger.info("Starting feature engineering")

        # -------------------------
        # DROP COLUMNS (EDA driven)
        # -------------------------
        drop_cols = [
            "Transaction ID",
            "Customer ID",
            "Transaction Date",
            "Customer Location",
            "IP Address",
            "Shipping Address",
            "Billing Address",
        ]
        df = df.drop(columns=drop_cols, errors="ignore")

        # -------------------------
        # FILTER AGE
        # -------------------------
        df = df[df["Customer Age"] >= 18]

        # -------------------------
        # LOG TRANSFORM
        # -------------------------
        df["Log_Transaction_Amount"] = self.log_transform(df["Transaction Amount"])

        # -------------------------
        # FEATURE ENGINEERING
        # -------------------------
        df["New_Account"] = (df["Account Age Days"] <= 30).astype(int)
        df["Early_Txn"] = df["Transaction Hour"].between(0, 5).astype(int)

        df["Age_Amount_Risk"] = (
            df["Customer Age"] * df["Log_Transaction_Amount"]
        )

        # -------------------------
        # ORDINAL ENCODING
        # -------------------------
        df["Quantity"] = df["Quantity"].astype(int)

        # -------------------------
        # ONE HOT ENCODING
        # -------------------------
        cat_cols = ["Device Used", "Product Category", "Payment Method"]
        df = pd.get_dummies(df, columns=cat_cols, drop_first=True)

        # -------------------------
        # DROP ORIGINAL AMOUNT
        # -------------------------

        return df

    def initiate_data_transformation(self) -> DataTransformationArtifact:
        try:
            logger.info("Starting data transformation phase")

            train_df = pd.read_csv(self.validation_artifact.valid_train_file_path)
            test_df = pd.read_csv(self.validation_artifact.valid_test_file_path)

            train_df = self.engineer_features(train_df)
            test_df = self.engineer_features(test_df)

            os.makedirs(self.config.data_transformation_dir, exist_ok=True)

            train_df.to_csv(self.config.transformed_train_path, index=False)
            test_df.to_csv(self.config.transformed_test_path, index=False)

            # Save feature engineering metadata (for inference parity)
            with open(self.config.preprocessing_object_path, "wb") as f:
                pickle.dump(
                    {
                        "columns": train_df.columns.tolist(),
                        "target": TARGET_COLUMN,
                    },
                    f,
                )

            logger.info("Data transformation completed successfully")
            logger.info(f"Transformed columns: {train_df.columns.tolist()} and target: {TARGET_COLUMN}")

            return DataTransformationArtifact(
                transformed_train_path=self.config.transformed_train_path,
                transformed_test_path=self.config.transformed_test_path,
                preprocessing_object_path=self.config.preprocessing_object_path,
            )

        except Exception as e:
            logger.error("Error during data transformation")
            raise CustomException(e, sys)
