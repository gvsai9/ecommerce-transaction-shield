import os
import sys
import yaml
import pickle
import mlflow
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import f1_score, classification_report
from imblearn.over_sampling import SMOTE

from src.logger import logger
from src.exception import CustomException
from src.constants.training_pipeline import TARGET_COLUMN
from src.entity.artifact_entity import DataTransformationArtifact, ModelTrainerArtifact
from src.entity.config_entity import ModelTrainerConfig


class ModelTrainer:
    def __init__(
        self,
        data_transformation_artifact: DataTransformationArtifact,
        model_trainer_config: ModelTrainerConfig,
    ):
        self.transformation_artifact = data_transformation_artifact
        self.config = model_trainer_config

    def load_data(self):
        train_df = pd.read_csv(self.transformation_artifact.transformed_train_path)
        test_df = pd.read_csv(self.transformation_artifact.transformed_test_path)

        X_train = train_df.drop(columns=[TARGET_COLUMN])
        y_train = train_df[TARGET_COLUMN]

        X_test = test_df.drop(columns=[TARGET_COLUMN])
        y_test = test_df[TARGET_COLUMN]

        return X_train, X_test, y_train, y_test

    def train_tree_model(self, X_train, X_test, y_train, y_test):
        logger.info("Training RandomForest (tree model)")

        model = RandomForestClassifier(
            n_estimators=200,
            max_depth=None,
            random_state=42,
            n_jobs=-1
        )

        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        score = f1_score(y_test, preds)

        return model, score

    def train_linear_model(self, X_train, X_test, y_train, y_test):
        logger.info("Training Logistic Regression (linear model)")

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        smote = SMOTE(random_state=42)
        X_train_bal, y_train_bal = smote.fit_resample(X_train_scaled, y_train)

        model = LogisticRegression(max_iter=1000)
        model.fit(X_train_bal, y_train_bal)

        preds = model.predict(X_test_scaled)
        score = f1_score(y_test, preds)

        return Pipeline([("scaler", scaler), ("model", model)]), score

    def initiate_model_training(self) -> ModelTrainerArtifact:
        try:
            logger.info("Starting model training phase")

            X_train, X_test, y_train, y_test = self.load_data()

            mlflow.set_experiment(experiment_id="0")

            results = {}

            with mlflow.start_run(run_name="tree_model"):
                tree_model, tree_score = self.train_tree_model(
                    X_train, X_test, y_train, y_test
                )
                mlflow.log_metric("f1_score", tree_score)
                results["RandomForest"] = (tree_model, tree_score)
                mlflow.log_param("model_type", "RandomForest")
                mlflow.log_param("data_version", self.transformation_artifact.transformed_train_path)


            with mlflow.start_run(run_name="linear_model"):
                linear_model, linear_score = self.train_linear_model(
                    X_train, X_test, y_train, y_test
                )
                mlflow.log_metric("f1_score", linear_score)
                results["LogisticRegression"] = (linear_model, linear_score)
                mlflow.log_param("model_type", "LogisticRegression")
                mlflow.log_param("data_version", self.transformation_artifact.transformed_train_path)


            best_model_name, (best_model, best_score) = max(
                results.items(), key=lambda x: x[1][1]
            )

            os.makedirs(self.config.model_trainer_dir, exist_ok=True)

            with open(self.config.trained_model_path, "wb") as f:
                pickle.dump(best_model, f)

            with open(self.config.metrics_file_path, "w") as f:
                yaml.dump(
                    {
                        "best_model": best_model_name,
                        "best_f1_score": best_score,
                    },
                    f,
                )

            logger.info(
                f"Best model: {best_model_name} | F1 Score: {best_score}"
            )

            return ModelTrainerArtifact(
                trained_model_path=self.config.trained_model_path,
                best_model_name=best_model_name,
                best_model_score=best_score,
            )

        except Exception as e:
            logger.error("Model training failed")
            raise CustomException(e, sys)
