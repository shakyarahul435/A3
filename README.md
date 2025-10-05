# A3 - Car Price Prediction Model

## Overview
A3 is a machine learning project for predicting used car prices. The model is trained, versioned, and deployed using MLflow, with automated CI/CD integration to the ML Brain Lab MLflow server.

**MLflow Server**: [https://mlflow.ml.brain.cs.ait.ac.th/](https://mlflow.ml.brain.cs.ait.ac.th/)

## Model Information
- **Model Name**: `st125982-a3-model`
- **Task**: Regression (Car Price Prediction)
- **Target Variable**: Selling Price
- **MLflow Experiment**: `st125982-a3`

## Features
The model uses the following input features:

| Feature | Type | Description |
|---------|------|-------------|
| `brand` | string | Car manufacturer (e.g., "Maruti") |
| `year` | integer | Manufacturing year |
| `engine` | integer | Engine capacity (cc) |
| `max_power` | float | Maximum power (bhp) |

### Example Input
```python
sample = pd.Series({
    'brand': 'Maruti',
    'year': 2014,
    'engine': 1248,
    'max_power': 74
})
```

## Project Structure
```
A3/
├── model/
│   └── st125982-a3-model.pkl      # Trained model artifact
├── src/
│   └── src.py                      # Source code utilities
├── app.py                          # Main application (Flask/Dash)
├── deploy_model.py                 # MLflow deployment script
├── test.py                         # Model testing suite
└── README.md                       # This file
```

## Installation

### Prerequisites
- Python 3.8+
- MLflow
- pandas
- scikit-learn
- joblib

### Setup
```bash
pip install mlflow pandas scikit-learn joblib flask dash
```

## Usage

### 1. Running the Application
The main application runs on port 8050:

```bash
python app.py
```

Access the application at: `http://0.0.0.0:8050`

### 2. Testing the Model
Run automated tests to verify model functionality:

```bash
python test.py
```

**Tests Performed:**
- ✓ Model accepts correct input format
- ✓ Output shape validation (ensures predictions match input length)

### 3. Deploying to MLflow
Deploy the model to the MLflow tracking server:

```bash
python deploy_model.py
```

This script:
1. Loads the local model from `model/st125982-a3-model.pkl`
2. Wraps it in MLflow's `pyfunc` format
3. Logs the model to the MLflow experiment
4. Registers it in the model registry

## Model Deployment Details

### MLflow Configuration
```python
mlflow.set_tracking_uri("https://mlflow.ml.brain.cs.ait.ac.th/")
mlflow.set_experiment("st125982-a3")
```

### Model Wrapper
The model uses a custom `CarPriceWrapper` class that implements MLflow's `PythonModel` interface for seamless deployment and serving.

## CI/CD Integration
Every new best model is automatically:
- Trained and evaluated
- Logged to MLflow with metrics and artifacts
- Registered in the model registry
- Versioned for reproducibility

## Loading the Model

### Local Loading
```python
import joblib

model = joblib.load("model/st125982-a3-model.pkl")
prediction = model.predict(sample_df)
```

### MLflow Loading
```python
import mlflow

model = mlflow.pyfunc.load_model("models:/st125982-a3-model/latest")
prediction = model.predict(sample_df)
```

## Model Performance
Check the MLflow UI for detailed metrics, parameters, and experiment tracking:
[https://mlflow.ml.brain.cs.ait.ac.th/](https://mlflow.ml.brain.cs.ait.ac.th/)

## Development

### Adding New Features
1. Update feature preprocessing in `src/src.py`
2. Retrain the model
3. Run tests with `python test.py`
4. Deploy with `python deploy_model.py`

### Running Tests
```bash
python test.py
```
Exit codes:
- `0`: All tests passed
- `1`: Some tests failed

## Contact
- **Student ID**: st125982
- **Project**: A3 - Car Price Prediction
- **MLflow Experiment**: st125982-a3

## License
Academic project for ML Brain Lab, AIT