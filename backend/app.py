from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import os
import pandas as pd
import numpy as np

app = Flask(__name__)
CORS(app)

# Load the trained model
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'lifespan_advanced.joblib')
try:
    model = joblib.load(MODEL_PATH)
    print(f"Model loaded successfully from {MODEL_PATH}")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "healthy", "model_loaded": model is not None})

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({"error": "Model not loaded on server"}), 500
    
    data = request.json
    try:
        # Preprocess input data to match training features
        # Features: age, gender, bmi, exercise, smoking, alcohol, systolic, cholesterol, glucose
        df = pd.DataFrame([{
            'age': float(data.get('age', 40)),
            'gender': 1 if str(data.get('gender')).lower() in ['male', '1'] else 0,
            'bmi': float(data.get('bmi', 25)),
            'exercise': int(data.get('exercise_level', data.get('exercise', 0))),
            'smoking': 1 if data.get('smoking') in ['yes', 1, True] else 0,
            'alcohol': 1 if data.get('alcohol') in ['yes', 1, True] else 0,
            'systolic': float(data.get('blood_pressure', data.get('systolic', 120))),
            'cholesterol': float(data.get('cholesterol', 200)),
            'glucose': float(data.get('glucose', 90))
        }])
        
        prediction = model.predict(df)[0]
        
        # Calculate feature importance/contribution (Simplified SHAP-like)
        # For RF, we can show how much each feature changed the prediction from the baseline
        # But for now, we'll return a simple response
        
        return jsonify({
            "prediction": float(np.round(prediction, 1)),
            "model_version": "RandomForest-v1.0",
            "confidence_interval": [float(np.round(prediction - 3, 1)), float(np.round(prediction + 3, 1))]
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == '__main__':
    app.run(port=5001, debug=True)
