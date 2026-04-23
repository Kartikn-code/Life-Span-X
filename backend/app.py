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

@app.route('/', methods=['GET'])
def index():
    return jsonify({
        "message": "LifeSpanX Neural API is running",
        "endpoints": ["/predict (POST)", "/health (GET)"],
        "status": "online"
    })

@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "healthy", "model_loaded": model is not None})

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({"error": "Model not loaded on server"}), 500
    
    data = request.json
    try:
        # Features: age, gender, bmi, exercise, smoking, alcohol, systolic, cholesterol, glucose
        feature_names = ['age', 'gender', 'bmi', 'exercise', 'smoking', 'alcohol', 'systolic', 'cholesterol', 'glucose']
        
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
        
        # Ensure correct column order
        df = df[feature_names]
        
        prediction = model.predict(df)[0]
        
        # Extract feature importance from the pipeline's model
        rf_model = model.named_steps['model']
        importances = rf_model.feature_importances_
        
        # Map back to names and scale for UI impact (heuristic)
        # RF importance is always positive, so we use the direction from Phase 1 linear model logic
        # Or just return the weights and let the UI handle it.
        importance_list = []
        for name, imp in zip(feature_names, importances):
            importance_list.append({
                "name": name,
                "weight": float(imp)
            })

        return jsonify({
            "prediction": float(np.round(prediction, 1)),
            "model_version": "RandomForest-v1.1",
            "feature_importance": importance_list,
            "confidence_interval": [float(np.round(prediction - 2.5, 1)), float(np.round(prediction + 2.5, 1))]
        })
        
    except Exception as e:
        print(f"Prediction Error: {e}")
        return jsonify({"error": str(e)}), 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True)
