from datetime import datetime
import os
from src.constants import training_pipeline

class TrainingPipelineConfig:
    def __init__(self, timestamp: str = None):
        if timestamp is None:
            timestamp = datetime.now().strftime("%m_%d_%Y_%H_%M_%S")

        self.pipeline_name = training_pipeline.PIPELINE_NAME
        self.artifact_root = training_pipeline.ARTIFACT_DIR
        self.artifact_dir = os.path.join(self.artifact_root, timestamp)
        self.model_dir = os.path.join(self.artifact_dir, training_pipeline.SAVED_MODEL_DIR)
        self.timestamp = timestamp




from src.constants import data_ingestion

class DataIngestionConfig:
    def __init__(self, training_pipeline_config: TrainingPipelineConfig):
        self.data_ingestion_dir = os.path.join(
            training_pipeline_config.artifact_dir,
            data_ingestion.DATA_INGESTION_DIR_NAME
        )

        self.raw_data_path = os.path.join(
            data_ingestion.DATA_INGESTION_RAW_DIR,
            data_ingestion.DATA_INGESTION_RAW_FILE_NAME
        )

        self.train_file_path = os.path.join(
            self.data_ingestion_dir,
            data_ingestion.DATA_INGESTION_TRAIN_FILE_NAME
        )

        self.test_file_path = os.path.join(
            self.data_ingestion_dir,
            data_ingestion.DATA_INGESTION_TEST_FILE_NAME
        )

from src.constants import data_validation

class DataValidationConfig:
    def __init__(self, training_pipeline_config):
        self.data_validation_dir = os.path.join(
            training_pipeline_config.artifact_dir,
            data_validation.DATA_VALIDATION_DIR_NAME
        )

        self.valid_data_dir = os.path.join(
            self.data_validation_dir,
            data_validation.DATA_VALIDATION_VALID_DIR
        )

        self.invalid_data_dir = os.path.join(
            self.data_validation_dir,
            data_validation.DATA_VALIDATION_INVALID_DIR
        )

        self.valid_train_file_path = os.path.join(
            self.valid_data_dir,
            training_pipeline.TRAIN_FILE_NAME
        )

        self.valid_test_file_path = os.path.join(
            self.valid_data_dir,
            training_pipeline.TEST_FILE_NAME
        )

        self.invalid_train_file_path = os.path.join(
            self.invalid_data_dir,
            training_pipeline.TRAIN_FILE_NAME
        )

        self.invalid_test_file_path = os.path.join(
            self.invalid_data_dir,
            training_pipeline.TEST_FILE_NAME
        )

        self.drift_report_file_path = os.path.join(
            self.data_validation_dir,
            data_validation.DATA_VALIDATION_DRIFT_REPORT_DIR,
            data_validation.DATA_VALIDATION_DRIFT_REPORT_FILE_NAME
        )

from src.constants import data_transformation

class DataTransformationConfig:
    def __init__(self, training_pipeline_config):
        self.data_transformation_dir = os.path.join(
            training_pipeline_config.artifact_dir,
            data_transformation.DATA_TRANSFORMATION_DIR_NAME
        )

        self.transformed_train_path = os.path.join(
            self.data_transformation_dir,
            data_transformation.TRANSFORMED_TRAIN_FILE_NAME
        )

        self.transformed_test_path = os.path.join(
            self.data_transformation_dir,
            data_transformation.TRANSFORMED_TEST_FILE_NAME
        )

        self.preprocessing_object_path = os.path.join(
            self.data_transformation_dir,
            data_transformation.PREPROCESSING_OBJECT_FILE_NAME
        )
from src.constants import model_trainer

class ModelTrainerConfig:
    def __init__(self, training_pipeline_config):
        self.model_trainer_dir = os.path.join(
            training_pipeline_config.artifact_dir,
            model_trainer.MODEL_TRAINER_DIR_NAME
        )

        self.trained_model_path = os.path.join(
            self.model_trainer_dir,
            model_trainer.MODEL_FILE_NAME
        )

        self.metrics_file_path = os.path.join(
            self.model_trainer_dir,
            model_trainer.METRICS_FILE_NAME
        )
from src.constants import model_evaluation

class ModelEvaluationConfig:
    def __init__(self, training_pipeline_config):
        self.model_evaluation_dir = os.path.join(
            training_pipeline_config.artifact_dir,
            model_evaluation.MODEL_EVALUATION_DIR_NAME
        )

        self.evaluation_report_path = os.path.join(
            self.model_evaluation_dir,
            model_evaluation.EVALUATION_REPORT_FILE_NAME
        )
