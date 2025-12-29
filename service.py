import bentoml
import pandas as pd

# 1. Load model reference (metadata only)
model_ref = bentoml.sklearn.get("fraud_detector:latest")

THRESHOLD = model_ref.info.metadata.get("threshold", 0.15)

# 🔒 Ensure target is NEVER in features
FEATURES = [
    col for col in model_ref.info.metadata["features"]
    if col != "Is Fraudulent"
]

@bentoml.service(
    name="fraud_detection_service",
    traffic={"timeout": 60}
)
class FraudService:
    bento_model = model_ref

    def __init__(self):
        self.model = self.bento_model.load_model()

    @bentoml.api
    def predict(self, input_data: dict) -> dict:
        df = pd.DataFrame([input_data])

        # 🛡️ Defensive: drop target if it ever appears
        df = df.drop(columns=["Is Fraudulent"], errors="ignore")

        # Align to training feature space
        df = df.reindex(columns=FEATURES, fill_value=0)

        prob = self.model.predict_proba(df)[:, 1][0]
        prediction = int(prob >= THRESHOLD)

        return {
            "fraud_probability": float(prob),
            "threshold": THRESHOLD,
            "is_fraud": prediction
        }
