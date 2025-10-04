import mlflow
import pickle
import pandas as pd
from src.src import *
import joblib
# Load local model

local_path = "model/st125982-a3-model.pkl"

predictor = joblib.load(local_path)
# Wrap model for MLflow
class CarPriceWrapper(mlflow.pyfunc.PythonModel):
    def __init__(self, predictor):
        self.predictor = predictor

    def predict(self, context, model_input: pd.DataFrame):
        return self.predictor.predict(model_input)

# Example input for MLflow schema
sample = pd.Series({
    'brand': 20,
    'year': 2014,
    'engine': 1248,
    'max_power': 74
})

sample_df = pd.DataFrame([sample])

mlflow.set_tracking_uri("https://mlflow.ml.brain.cs.ait.ac.th/")
mlflow.set_experiment("st125982-a3")

with mlflow.start_run(run_name="logistic_regression_deploy") as run:
    mlflow.pyfunc.log_model(
        name="model",
        python_model=CarPriceWrapper(predictor),
        input_example=sample_df
    )
    # Construct proper model URI to register
    model_uri = f"runs:/{run.info.run_id}/model"

# Register as a new version
registered_model = mlflow.register_model(
    model_uri=model_uri,
    name="st125982-a3-model"
)

print("Model deployed to MLflow!")