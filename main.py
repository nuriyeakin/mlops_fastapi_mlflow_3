from fastapi import FastAPI, Depends, Request
from schemas import hepsiburada
import os
from sqlalchemy.orm import Session
from mlflow.sklearn import load_model

# Tell where is the tracking server and artifact server
os.environ['MLFLOW_TRACKING_URI'] = 'http://localhost:5000/'
os.environ['MLFLOW_S3_ENDPOINT_URL'] = 'http://localhost:9000/'


# Learn, decide and get model from mlflow model registry
model_name = "hepsiburadaRFModel"
model_version = 1
model = load_model(
    model_uri=f"models:/{model_name}/{model_version}"
)

app = FastAPI()

def make_hepsiburada_prediction(model, request):
    # parse input from request
    memory= request["memory"]
    ram= request["ram"]
    screen_size= request["screen_size"]
    power= request["power"]
    front_camera= request["front_camera"]
    rc1= request["rc1"]
    rc3= request["rc3"]
    rc5= request["rc5"]
    rc7= request["rc7"]


    # Make an input vector
    hepsiburada = [[memory, ram, screen_size, power, front_camera, rc1, rc3, rc5, rc7]]

    # Predict
    prediction = model.predict(hepsiburada)

    return prediction[0]

# Advertising Prediction endpoint
@app.post("/prediction/hepsiburada")
def predict_hepsiburada(request: hepsiburada):
    prediction = make_hepsiburada_prediction(model, request.dict())
    return prediction
