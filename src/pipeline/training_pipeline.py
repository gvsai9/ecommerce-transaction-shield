from src.logger import logger
from src.entity.config_entity import TrainingPipelineConfig
from src.entity.config_entity import DataIngestionConfig
from src.components.data_ingestion import DataIngestion


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

        logger.info("Training pipeline completed")
