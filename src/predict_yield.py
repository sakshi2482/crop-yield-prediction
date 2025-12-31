import pickle
import numpy as np
from pathlib import Path

def predict_crop_yield(rainfall, temperature, soil_ph):
    BASE_DIR = Path(__file__).resolve().parent
    model_path = BASE_DIR / "crop_yield_model.pkl"
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    input_data = np.array([[rainfall, temperature, soil_ph]])
    prediction = model.predict(input_data)
    return prediction[0]

if __name__ == "__main__":
    print("🌾 Crop Yield Prediction System")

    rainfall = float(input("Enter rainfall (mm): "))
    temperature = float(input("Enter temperature (°C): "))
    soil_ph = float(input("Enter soil pH: "))

    result = predict_crop_yield(rainfall, temperature, soil_ph)

    print("\n✅ Predicted Crop Yield:", round(result, 2))
