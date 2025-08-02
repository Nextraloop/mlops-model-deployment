from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import logging
import os

app = FastAPI()

model_path = "models/best_model.pkl"
if not os.path.exists(model_path):
    raise Exception(f"Model file not found at {model_path}")

model = joblib.load(model_path)

logging.basicConfig(filename='logs/predictions.log', level=logging.INFO, format='%(asctime)s - %(message)s')

class InputData(BaseModel):
    features: list

@app.post("/predict")
def predict(data: InputData):
    prediction = model.predict([data.features])
    logging.info(f"Input: {data.features}, Prediction: {prediction[0]}")
    return {"prediction": float(prediction[0])}
