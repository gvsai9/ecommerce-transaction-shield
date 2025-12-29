import bentoml
import pickle
import yaml

MODEL_PATH = "artifacts/latest/model_trainer/model.pkl"
EVAL_PATH = "artifacts/latest/model_evaluation/evaluation.yaml"
PREPROCESS_PATH = "artifacts/latest/data_transformation/feature_engineering.pkl"

with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

with open(PREPROCESS_PATH, "rb") as f:
    preprocess_meta = pickle.load(f)

with open(EVAL_PATH) as f:
    eval_report = yaml.safe_load(f)

bentoml.sklearn.save_model(
    "fraud_detector",
    model,
    metadata={
        "threshold": eval_report.get("best_threshold", 0.15),
        "features": preprocess_meta["columns"],
    },
)
