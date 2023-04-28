import time
import pickle
import os
import random

import pandas as pd
import numpy as np
from flask import Flask, jsonify, request, make_response
from prometheus_client import Counter, Histogram, Gauge, generate_latest
from sqlalchemy import create_engine, text
from datetime import datetime, timedelta

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

app = Flask(__name__)

REQUEST_COUNT = Counter(
    "ml_request_count",
    "App Request Count",
    ["prometheus_app", "method", "endpoint", "http_status"],
)

REQUEST_LATENCY = Histogram(
    "ml_request_latency_seconds", "Request latency", ["app_name", "endpoint"]
)

CONTENT_TYPE_LATEST = str("text/plain; version=0.0.4; charset=utf-8")


MODEL_TYPE = Counter(name="model_type_request_count", labelnames=["model_type"], documentation="Model type")

features = ["crim", "zn", "indus", "chas", "nox", "rm", "age", "dis", "rad", "tax", "ptratio", "b", "lstat"]
target = "medv"

INPUT_DATA_METRICS = Gauge(name="model_input_data", 
                           labelnames=["feature"],
                           documentation="Model Input Data")
EXPLAINABILITY_METRICS = Gauge(name="model_features_explainability",
                               labelnames=["feature"],
                               documentation="Model Features Explainability")

PREDICTED_VALUE = Gauge(name="MEDV", documentation="Median value of owner-occupied homes in $1000's")

ML_PERFORMANCES = Gauge(name="ml_performance", documentation="RMSE", labelnames=["model"])

@app.before_request
def before_request():
    request.start_time = time.time()


@app.after_request
def after_request(response):
    resp_time = time.time() - request.start_time
    REQUEST_COUNT.labels(
        "prometheus_app", request.method, request.path, response.status_code
    ).inc()
    REQUEST_LATENCY.labels("prometheus_app", request.path).observe(resp_time)
    return response


@app.route("/")
def home():
    return make_response(jsonify({"status": "ok"}), 200)

@app.route("/train", methods=['GET', 'POST'])
def train():

    # Define the PostgreSQL connection URL
    db_url = f"postgresql://{os.environ['POSTGRES_USER']}:{os.environ['POSTGRES_PASSWORD']}@db:5432/{os.environ['POSTGRES_DB']}"
    engine = create_engine(db_url)

    # Query the data from the boston table
    query = "SELECT * FROM boston_dataset"
    df = pd.read_sql(query, engine)
    df.drop("home_id", axis=1, inplace=True)

    # Split the dataframe into data and target
    data = df.drop('medv', axis=1).values
    target = df['medv'].values

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.2, random_state=42)
    
    if not os.path.exists('models'):
        os.mkdir('models')
        
    performances = {}
    for regressor_module in [GradientBoostingRegressor, MLPRegressor]:
        regressor_model = regressor_module()
        
        # Train model
        regressor_model.fit(X_train, y_train)

        # Make predictions on the test set
        y_pred = regressor_model.predict(X_test)

        # Compute the mean squared error on the test set
        mse = mean_squared_error(y_test, y_pred)
        print(f"Mean Squared Error {regressor_module.__name__}:", mse)
        performances[regressor_module.__name__] = mse

        # Serialize the model
        with open(f'models/{regressor_module.__name__}.pkl', 'wb') as fp:
            pickle.dump(obj=regressor_model, file=fp)

    return {"mean_squared_error": performances}

@app.route("/predict", methods=['GET'])
def predict():
    data = request.get_json()
    home_id = data.pop('home_id')

    X = np.array([data[f] for f in features])
    for feat, val in zip(features, X):
        print(feat, val)
        INPUT_DATA_METRICS.labels(feature=feat).set(value=val)

    # Random Model Chosen
    model_type = random.choice([GradientBoostingRegressor, MLPRegressor])
    model_name = model_type.__name__.split(".")[-1]
    
    MODEL_TYPE.labels(model_type.__name__).inc()

    # Random Prediction
    with open(f'models/{model_name}.pkl', 'rb') as fp:
        ml_model = pickle.load(file=fp)

    prediction = ml_model.predict(X.reshape(1, -1))[0]
    PREDICTED_VALUE.set(value=prediction)

    # Random Explainabilities
    expl = {f: random.random() for f in features}
    sum_ran = sum(list(expl.values()))
    expl_norm = {f: v / sum_ran for f, v in expl.items()}
    for feat, val in expl_norm.items():
        EXPLAINABILITY_METRICS.labels(feature=feat).set(value=val)

    # Log the prediction in the database
    # Define the PostgreSQL connection URL
    db_url = f"postgresql://{os.environ['POSTGRES_USER']}:{os.environ['POSTGRES_PASSWORD']}@db:5432/{os.environ['POSTGRES_DB']}"
    engine = create_engine(db_url)

    # Create a SQL query that inserts the data into the table only if the ID doesn't exist
    insert_query = f"""
        INSERT INTO ml_predictions (event_time, home_id, prediction, model_used)
        SELECT CURRENT_TIMESTAMP, '{home_id}', {prediction}, '{model_name}';
    """
    
    # Execute the SQL query using the engine
    with engine.connect() as conn:
        conn.execute(text(insert_query))
        conn.commit()

    return {"prediction": prediction}

@app.route("/metrics")
def metrics():

    # Define the PostgreSQL connection URL
    db_url = f"postgresql://{os.environ['POSTGRES_USER']}:{os.environ['POSTGRES_PASSWORD']}@db:5432/{os.environ['POSTGRES_DB']}"
    engine = create_engine(db_url)
    PROMETHEUS_SCRAPE_INTERVAL = os.environ["PROMETHEUS_SCRAPE_INTERVAL"] # in seconds
    last_scrape_time = datetime.now() - timedelta(seconds=int(PROMETHEUS_SCRAPE_INTERVAL))
    print("Prometheus Scrape Interval:", PROMETHEUS_SCRAPE_INTERVAL)
    print("Last Scrape Time Believed:", last_scrape_time)

    query = f"""
            SELECT  ML_TB.event_time, 
                    ML_TB.home_id, 
                    ACTUAL_TB.actual, 
                    ML_TB.prediction, 
                    ML_TB.model_used
            FROM ml_predictions ML_TB
            INNER JOIN  (SELECT home_id, medv AS ACTUAL
                        FROM boston_dataset) ACTUAL_TB
            ON  ML_TB.home_id::text = ACTUAL_TB.home_id::text
                AND event_time > '{last_scrape_time.strftime('%Y-%m-%d %H:%M:%S')}'
            """

    from sklearn.metrics import mean_squared_error
    df = pd.read_sql(query, engine)
    print("Predictions/Actuals Since Last Scrape:\n", df.head())
    if len(df):
        rmse_series = df.groupby("model_used").apply(lambda x: mean_squared_error(x["actual"], x["prediction"], squared=False))
        for k, v in rmse_series.items():
            ML_PERFORMANCES.labels(model=k).set(value=v)
    else:
        rmse_series = None
    
    return generate_latest()
