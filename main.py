from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

# Load your trained model
model = joblib.load("earthquake_damage_model.pkl")

# Initialize FastAPI app
app = FastAPI(title="Earthquake Damage Prediction API")

# Define the input schema using Pydantic
class EarthquakeInput(BaseModel):
    latitude: float
    longitude: float
    depth: float
    magnitude: float

# Mapping from prediction class to label (adjust if needed)
label_map = {
    0: "Slight damage",
    1: "Moderate damage",
    2: "Major damage",
    3: "Great damage"
}

@app.post("/predict/")
def predict_damage(data: EarthquakeInput):
    # Prepare input as numpy array
    input_data = np.array([
        data.latitude,
        data.longitude,
        data.depth,
        data.magnitude
    ]).reshape(1, -1)

    # Predict class
    prediction = model.predict(input_data)[0]
    label = label_map.get(prediction, "Unknown")

    return {
        "prediction_class": int(prediction),
        "damage_description": label
    }
