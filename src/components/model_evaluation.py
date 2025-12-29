import os
import sys
import yaml
import pickle
import pandas as pd

from sklearn.metrics import (
    fbeta_score,
    precision_score,
    recall_score
)

from src.logger import logger
from src.exception import CustomException
from src.constants.model_evaluation import (
    MIN_F2_SCORE,
    MIN_RECALL,
    MIN_PRECISION
)
from src.constants.training_pipeline import TARGET_COLUMN
from src.entity.artifact_entity import (
    ModelTrainerArtifact,
    DataTransformationArtifact,
    ModelEvaluationArtifact
)
from src.entity.config_entity import ModelEvaluationConfig


class ModelEvaluation:
    def __init__(
        self,
        model_trainer_artifact: ModelTrainerArtifact,
        data_transformation_artifact: DataTransformationArtifact,
        model_evaluation_config: ModelEvaluationConfig,
    ):
        self.model_trainer_artifact = model_trainer_artifact
        self.data_transformation_artifact = data_transformation_artifact
        self.config = model_evaluation_config

    def initiate_model_evaluation(self) -> ModelEvaluationArtifact:
        try:
            logger.info("Starting model evaluation phase")

            # Load model
            with open(self.model_trainer_artifact.trained_model_path, "rb") as f:
                model = pickle.load(f)

            # Load test data
            test_df = pd.read_csv(self.data_transformation_artifact.transformed_test_path)
            X_test = test_df.drop(columns=[TARGET_COLUMN])
            y_test = test_df[TARGET_COLUMN]

            # Predict using probability + tuned threshold
            probs = model.predict_proba(X_test)[:, 1]
            threshold = 0.15  # later load from artifact
            preds = (probs >= threshold).astype(int)

            # Metrics
            f2 = fbeta_score(y_test, preds, beta=2)
            recall = recall_score(y_test, preds)
            precision = precision_score(y_test, preds)

            logger.info(f"F2={f2}, Recall={recall}, Precision={precision}")

            # Guardrails
            is_accepted = (
                f2 >= MIN_F2_SCORE
                and recall >= MIN_RECALL
                and precision >= MIN_PRECISION
            )

            os.makedirs(self.config.model_evaluation_dir, exist_ok=True)

            report = {
                "f2_score": float(f2),
                "recall": float(recall),
                "precision": float(precision),
                "accepted": is_accepted,
            }

            with open(self.config.evaluation_report_path, "w") as f:
                yaml.dump(report, f)

            if not is_accepted:
                logger.warning("Model rejected by evaluation guardrails")

            return ModelEvaluationArtifact(
                is_model_accepted=is_accepted,
                evaluated_metric=f2,
                evaluation_report_path=self.config.evaluation_report_path,
            )

        except Exception as e:
            raise CustomException(e, sys)
