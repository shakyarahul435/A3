import dash
from dash import html, dcc, Input, Output, State
import pandas as pd
import mlflow.pyfunc
import os


os.environ["MLFLOW_TRACKING_USERNAME"]="admin"
os.environ["MLFLOW_TRACKING_PASSWORD"]="password"
# Load your MLflow model


mlflow_tracking_url = os.environ.get("MLFLOW_TRACKING_URL", "https://mlflow.ml.brain.cs.ait.ac.th/")

mlflow.set_tracking_uri(mlflow_tracking_url)
mlflow.set_experiment("st125982-a3")


model_name = "st125982-a3-model" 


# Sample input data
sample = pd.Series({
    'brand': 20,
    'year': 2014,
    'engine': 1248,
    'max_power': 74
})

# Initialize Dash app
app = dash.Dash(__name__)

app.layout = html.Div([
    html.H1("Car Price Class Prediction"),
    
    # html.Label("Brand"),
    # dcc.Input(id='brand', type='number', value=sample['brand']),

    html.Div([
    html.Div([
    html.Label(["Car Brand Name:"],style={"fontWeight": "bold"}),   # using label for dropdown 
    dcc.Dropdown(
        id="brand",
        options=[   # Provided with Brand names and value as per LabelEncoder in A_Z ascending order
            {"label": "Ambassador", "value": 0},
            {"label": "Ashok", "value": 1},
            {"label": "Audi", "value": 2},
            {"label": "BMW", "value": 3},
            {"label": "Chevrolet", "value": 4},
            {"label": "Daewoo", "value": 5},
            {"label": "Datsun", "value": 6},
            {"label": "Fiat", "value": 7},
            {"label": "Force", "value": 8},
            {"label": "Ford", "value": 9},
            {"label": "Honda", "value": 10},
            {"label": "Hyundai", "value": 11},
            {"label": "Isuzu", "value": 12},
            {"label": "Jaguar", "value": 13},
            {"label": "Jeep", "value": 14},
            {"label": "Kia", "value": 15},
            {"label": "Land", "value": 16},
            {"label": "Lexus", "value": 17},
            {"label": "MG", "value": 18},
            {"label": "Mahindra", "value": 19},
            {"label": "Maruti", "value": 20},
            {"label": "Mercedes-Benz", "value": 21},
            {"label": "Mitsubishi", "value": 22},
            {"label": "Nissan", "value": 23},
            {"label": "Opel", "value": 24},
            {"label": "Peugeot", "value": 25},
            {"label": "Renault", "value": 26},
            {"label": "Skoda", "value": 27},
            {"label": "Tata", "value": 28},
            {"label": "Toyota", "value": 29},
            {"label": "Volkswagen", "value": 30},
            {"label": "Volvo", "value": 31}
        ],
        placeholder="Select Brand",
        value=sample['brand'],
        style={"margin-bottom": "10px","width": "22rem"}
        ),
    ],style={"margin-top": "1rem"}),
    ],style={"display": "inline-block","verticalAlign": "top","margin-right": "2rem"}),
    
    html.Label("Year"),
    dcc.Input(id='year', type='number', value=sample['year']),
    
    html.Label("Engine (cc)"),
    dcc.Input(id='engine', type='number', value=sample['engine']),
    
    html.Label("Max Power (HP)"),
    dcc.Input(id='max_power', type='number', value=sample['max_power']),
    
    html.Button("Predict", id='predict-btn', n_clicks=0),
    
    html.H2("Prediction:"),
    dcc.Loading(
        id="loading-prediction",
        type="circle",  # you can use "default", "dot", or "circle"
        children=html.Div(id='prediction-output')
    ),

])

@app.callback(
    Output('prediction-output', 'children'),
    Input('predict-btn', 'n_clicks'),
    State('brand', 'value'),
    State('year', 'value'),
    State('engine', 'value'),
    State('max_power', 'value')
)
def predict(n_clicks, brand, year, engine, max_power):
    # model = mlflow.pyfunc.load_model(f"models:/{model_name}/latest")
    try:
        model = mlflow.pyfunc.load_model(f"models:/{model_name}/latest")
    except Exception as e:
        print("Failed to load model:", e)
        model = None

    if n_clicks > 0:
        input_df = pd.DataFrame([{
            'brand': brand,
            'year': year,
            'engine': engine,
            'max_power': max_power
        }])
        prediction = model.predict(input_df)
        return f"Predicted Price Class: {prediction[0]}"
    return "No prediction yet."

if __name__ == '__main__':
    # app.run_server(debug=True, host='0.0.0.0', port=8050)
    app.run(debug=True, host='0.0.0.0', port=8050)
