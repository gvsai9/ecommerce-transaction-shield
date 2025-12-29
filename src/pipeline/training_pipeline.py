from src.logger import logger
from src.entity.config_entity import TrainingPipelineConfig
from src.entity.config_entity import DataIngestionConfig
from src.components.data_ingestion import DataIngestion
from src.entity.config_entity import DataValidationConfig
from src.components.data_validation import DataValidation
from src.entity.config_entity import DataTransformationConfig
from src.components.data_transformation import DataTransformation
from src.entity.config_entity import ModelTrainerConfig
from src.components.model_trainer import ModelTrainer
from src.entity.config_entity import ModelEvaluationConfig
from src.components.model_evaluation import ModelEvaluation
from src.utils import update_latest_artifacts


class TrainingPipeline:
    def __init__(self):
        self.training_pipeline_config = TrainingPipelineConfig()

    def run_pipeline(self):
        logger.info("Training pipeline started")

        # Data Ingestion
        data_ingestion_config = DataIngestionConfig(
            training_pipeline_config=self.training_pipeline_config
        )

        data_ingestion = DataIngestion(data_ingestion_config)
        data_ingestion_artifact = data_ingestion.initiate_data_ingestion()

        logger.info(
            f"Data ingestion completed. "
            f"Train file: {data_ingestion_artifact.train_file_path}, "
            f"Test file: {data_ingestion_artifact.test_file_path}"
        )

        logger.info("Proceeding to Data Validation...")

               # =========================
            # DATA VALIDATION
            # =========================
        logger.info("Starting Data Validation")

        data_validation_config = DataValidationConfig(
            training_pipeline_config=self.training_pipeline_config
        )

        data_validation = DataValidation(
            data_ingestion_artifact=data_ingestion_artifact,
            data_validation_config=data_validation_config,
        )

        data_validation_artifact = data_validation.initiate_data_validation()

        if not data_validation_artifact.validation_status:
            raise Exception(
                "Data validation failed. Pipeline execution stopped."
            )

        logger.info(
            "Data validation completed successfully | "
            f"Drift report: {data_validation_artifact.drift_report_file_path}"
        )


        logger.info("Starting Data Transformation")

        data_transformation_config = DataTransformationConfig(
            training_pipeline_config=self.training_pipeline_config
        )

        data_transformation = DataTransformation(
            data_validation_artifact=data_validation_artifact,
            data_transformation_config=data_transformation_config,
        )

        data_transformation_artifact = data_transformation.initiate_data_transformation()

        logger.info(
            f"Data transformation completed | "
            f"Train features: {data_transformation_artifact.transformed_train_path}"
        )
        logger.info("Starting Model Training")

        model_trainer_config = ModelTrainerConfig(
            training_pipeline_config=self.training_pipeline_config
        )

        model_trainer = ModelTrainer(
            data_transformation_artifact=data_transformation_artifact,
            model_trainer_config=model_trainer_config,
        )

        model_trainer_artifact = model_trainer.initiate_model_training()

        logger.info(
            f"Model training completed | Best model: {model_trainer_artifact.best_model_name}"
        )
        
        logger.info("Starting Model Evaluation")

        model_evaluation_config = ModelEvaluationConfig(
            training_pipeline_config=self.training_pipeline_config
        )

        model_evaluation = ModelEvaluation(
            model_trainer_artifact=model_trainer_artifact,
            data_transformation_artifact=data_transformation_artifact,
            model_evaluation_config=model_evaluation_config,
        )

        model_evaluation_artifact = model_evaluation.initiate_model_evaluation()

        if not model_evaluation_artifact.is_model_accepted:
            raise Exception("Model rejected by evaluation guardrails")

        logger.info("Model accepted by evaluation")

        if model_evaluation_artifact.is_model_accepted:
            update_latest_artifacts(
                current_artifact_dir=self.training_pipeline_config.artifact_dir,
                latest_dir="artifacts/latest"
            )

