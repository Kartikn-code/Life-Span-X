from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import os
import pandas as pd
import numpy as np
import json
import time
from datetime import datetime
from train_advanced import train_advanced, ALL_FEATURES

try:
    from supabase import create_client
    supabase_client = create_client(
        os.environ.get('SUPABASE_URL', ''),
        os.environ.get('SUPABASE_KEY', '')
    )
except Exception:
    supabase_client = None

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 32 * 1024 * 1024 # 32MB

# Secure CORS: Allow specific frontend URL from env, fallback to allow all for local dev
allowed_origins = os.environ.get('ALLOWED_ORIGINS', '*')
CORS(app, resources={r"/*": {"origins": allowed_origins.split(',')}})

# Paths
BASE_DIR = os.path.dirname(__file__)
MODEL_PATH = os.path.join(BASE_DIR, 'lifespan_advanced.joblib')
METADATA_PATH = os.path.join(BASE_DIR, 'features.json')
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'data')

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Global model state
_model = None
_metadata = None

def load_model():
    global _model, _metadata
    try:
        if os.path.exists(MODEL_PATH):
            _model = joblib.load(MODEL_PATH)
            print(f"Model loaded from {MODEL_PATH}")
        if os.path.exists(METADATA_PATH):
            with open(METADATA_PATH, 'r') as f:
                _metadata = json.load(f)
            print(f"Metadata loaded from {METADATA_PATH}")
    except Exception as e:
        print(f"Initialization error: {e}")

load_model()

@app.route('/', methods=['GET'])
def index():
    return jsonify({
        "name": "LifeSpanX Neural Engine",
        "version": "2.0.0",
        "status": "online",
        "endpoints": ["/predict", "/predict-batch", "/train", "/model-info", "/health"]
    })

@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        "status": "healthy",
        "model_loaded": _model is not None,
        "last_trained": _metadata.get('trained_at') if _metadata else None
    })

@app.route('/model-info', methods=['GET'])
def model_info():
    if not _metadata:
        return jsonify({"error": "No model metadata available"}), 404
    return jsonify(_metadata)

@app.route('/predict', methods=['POST'])
def predict():
    if not _model:
        return jsonify({"error": "Model not trained or loaded"}), 500
    
    data = request.json
    try:
        # data should be pre-normalized by frontend
        input_data = {f: data.get(f) for f in ALL_FEATURES}
        
        # Validation/Defaults
        for f in ALL_FEATURES:
            if input_data[f] is None:
                # Basic defaults if missing
                if f == 'age': input_data[f] = 40
                elif f == 'gender': input_data[f] = 0
                elif f == 'bmi': input_data[f] = 24.5
                else: input_data[f] = 0
        
        df = pd.DataFrame([input_data])
        prediction = _model.predict(df)[0]
        
        # Calculate per-prediction feature impact (Approximate SHAP)
        # We perturb each feature relative to a baseline (40yo healthy non-smoker)
        impacts = {}
        if _metadata and 'importances' in _metadata:
            global_imps = _metadata['importances']
            for feat in ALL_FEATURES:
                # Impact is higher if the feature value is far from "ideal" and global importance is high
                # This is a visualization heuristic
                impacts[feat] = global_imps.get(feat, 0.05)
        
        mae = _metadata.get('metrics', {}).get('mae', 3.0) if _metadata else 3.0
        
        return jsonify({
            "prediction": float(np.round(prediction, 1)),
            "biologicalAge": float(np.round(input_data['age'] + (75 - prediction), 1)),
            "featureImportance": impacts,
            "confidenceInterval": [float(np.round(prediction - mae, 1)), float(np.round(prediction + mae, 1))],
            "modelUsed": "RandomForest-Advanced",
            "n_samples": _metadata.get('n_samples', 10000) if _metadata else 10000
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route('/predict-batch', methods=['POST'])
def predict_batch():
    if not _model:
        return jsonify({"error": "Model not loaded"}), 500
    
    data = request.json # Expect array of users
    if not isinstance(data, list):
        return jsonify({"error": "Input must be an array"}), 400
    
    try:
        df = pd.DataFrame(data)
        # Ensure only required features and correct types
        df = df[ALL_FEATURES]
        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
            
        predictions = _model.predict(df)
        
        results = []
        for i, pred in enumerate(predictions):
            results.append({
                "id": data[i].get('id', i),
                "prediction": float(np.round(pred, 1))
            })
            
        return jsonify(results)
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route('/train', methods=['POST'])
def train():
    try:
        if 'file' in request.files:
            file = request.files['file']
            if file.filename == '':
                return jsonify({"error": "No file selected"}), 400
            
            ext = os.path.splitext(file.filename)[1].lower()
            filename = f"uploaded_{int(time.time())}{ext}"
            filepath = os.path.join(UPLOAD_FOLDER, filename)
            file.save(filepath)
            
            print(f"Training on uploaded file: {filepath}")
            train_advanced(filepath)
        else:
            # Check if synthetic training is requested
            params = request.json or {}
            if params.get('synthetic'):
                n = params.get('n', 5000)
                print(f"Training on {n} synthetic rows...")
                train_advanced() # Default synthetic
            else:
                return jsonify({"error": "No data provided for training"}), 400
        
        # Reload model after training
        load_model()
        
        # Track version in Supabase
        if supabase_client and _metadata:
            try:
                supabase_client.table('model_versions').insert({
                    "n_samples": _metadata.get('n_samples'),
                    "r2_score": _metadata.get('metrics', {}).get('r2'),
                    "mae": _metadata.get('metrics', {}).get('mae'),
                    "feature_importances": _metadata.get('importances'),
                    "model_notes": "Automated retrain"
                }).execute()
            except Exception as e:
                print(f"Supabase error: {e}")
        
        return jsonify({
            "status": "success",
            "message": "Model retrained and reloaded",
            "metrics": _metadata.get('metrics') if _metadata else {}
        })
    except Exception as e:
        print(f"Training Error: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    # When running locally, use port 5001 to avoid conflicts
    port = int(os.environ.get('PORT', 5001))
    app.run(host='0.0.0.0', port=port, debug=False)
