import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import joblib
import os
import json
import argparse
from supabase import create_client, Client
from dotenv import load_dotenv

# Load env vars
load_dotenv()

# Feature Names and Types
NUMERIC_FEATURES = ['age', 'bmi', 'blood_pressure', 'cholesterol', 'glucose', 'sleep_hours', 'smoking', 'alcohol', 'exercise_level', 'stress_level']
BINARY_FEATURES = ['gender', 'heart_disease', 'diabetes', 'stroke']
ORDINAL_FEATURES = [] # All moved to numeric for smoothness
ALL_FEATURES = NUMERIC_FEATURES + BINARY_FEATURES + ORDINAL_FEATURES

# Supabase Setup
SUPABASE_URL = os.environ.get('SUPABASE_URL') or os.environ.get('VITE_SUPABASE_URL')
SUPABASE_KEY = os.environ.get('SUPABASE_KEY') or os.environ.get('VITE_SUPABASE_ANON_KEY')

def get_supabase():
    if SUPABASE_URL and SUPABASE_KEY:
        try:
            return create_client(SUPABASE_URL, SUPABASE_KEY)
        except: return None
    return None

def generate_realistic_data(n=10000):
    np.random.seed(42)
    age = np.clip(np.random.normal(52, 15, n), 18, 90)
    gender = np.random.choice([0, 1], n)
    bmi = np.clip(np.random.normal(26, 6, n), 15, 50)
    # Continuous Scales (0.0 to 1.0)
    exercise_level = np.random.uniform(0, 1, n)
    smoking = np.random.uniform(0, 1, n)
    alcohol = np.random.uniform(0, 1, n)
    stress_level = np.random.uniform(0, 1, n)
    blood_pressure = np.random.normal(125, 15, n)
    cholesterol = np.random.normal(190, 40, n)
    glucose = np.random.normal(95, 25, n)
    diabetes = np.random.choice([0, 1], n, p=[0.9, 0.1])
    heart_disease = np.random.choice([0, 1], n, p=[0.92, 0.08])
    stroke = np.random.choice([0, 1], n, p=[0.97, 0.03])
    sleep_hours = np.random.normal(7, 1, n)
    stress_level = np.random.randint(1, 6, n)
    
    # 🎯 HIGH-CONTRAST MEDICAL ENGINE (FIXED)
    y = 80.0 
    y -= (smoking * 15.0)           # -15 years
    y += (exercise_level * 5.0)     # +15 years max
    y -= (diabetes * 12.0)          # -12 years
    y -= (heart_disease * 14.0)     # -14 years
    y -= (stroke * 16.0)            # -16 years
    y -= ((blood_pressure - 120).clip(0) * 0.2)
    y -= (np.abs(bmi - 22) * 0.8)
    y += (1 - gender) * 4.0
    y += np.random.normal(0, 0.5, n) # Minimal noise
    y = np.clip(y, age + 1, 105)
    
    X = pd.DataFrame({
        'age': age, 'gender': gender, 'bmi': bmi, 'exercise_level': exercise_level,
        'smoking': smoking, 'alcohol': alcohol, 'blood_pressure': blood_pressure,
        'cholesterol': cholesterol, 'glucose': glucose, 'heart_disease': heart_disease,
        'diabetes': diabetes, 'stroke': stroke, 'sleep_hours': sleep_hours, 'stress_level': stress_level
    })
    return X, y

def train_advanced(data_path=None):
    supabase = get_supabase()
    backend_dir = os.path.dirname(__file__)
    data_dir = os.path.join(backend_dir, 'data')
    os.makedirs(data_dir, exist_ok=True)
    master_path = os.path.join(data_dir, 'master_training_data.csv')
    
    # Step 1: Get and Blend Data
    X_gen, y_gen = generate_realistic_data(5000)
    df_gen = X_gen.assign(lifespan=y_gen)
    
    if data_path and os.path.exists(data_path):
        print(f"📂 Blending uploaded data with medical baseline...")
        df_uploaded = pd.read_csv(data_path) if data_path.endswith('.csv') else pd.read_excel(data_path)
        # Ensure uploaded data has correct headers
        df_uploaded = map_headers(df_uploaded)
        df_train = pd.concat([df_uploaded, df_gen], ignore_index=True).dropna(subset=['lifespan'])
    else:
        print("🎲 Training on fresh High-Sensitivity medical baseline...")
        df_train = df_gen

    # Step 2: Preprocess
    if 'lifespan' not in df_train.columns:
        df_train['lifespan'] = 78.0
        
    X = df_train[ALL_FEATURES]
    y = df_train['lifespan']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Step 3: Train
    preprocessor = ColumnTransformer(transformers=[
        ('num', StandardScaler(), NUMERIC_FEATURES),
        ('pass', 'passthrough', BINARY_FEATURES + ORDINAL_FEATURES)
    ])
    model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
    pipeline = Pipeline([('preprocessor', preprocessor), ('model', model)])
    
    print(f"🧠 Training Brain on {len(X)} records...")
    pipeline.fit(X_train, y_train)
    
    # Step 4: Evaluate
    r2 = r2_score(y_test, pipeline.predict(X_test))
    mae = mean_absolute_error(y_test, pipeline.predict(X_test))
    print(f"✨ Intelligence (R²): {r2:.4f}")
    
    # Step 5: Save (FIXED)
    joblib.dump(pipeline, os.path.join(backend_dir, 'lifespan_advanced.joblib'))
    
    # Correct names mapping for ColumnTransformer order
    ordered_features = NUMERIC_FEATURES + BINARY_FEATURES + ORDINAL_FEATURES
    feat_imp = dict(zip(ordered_features, pipeline.named_steps['model'].feature_importances_))
    
    with open(os.path.join(backend_dir, 'features.json'), 'w') as f:
        json.dump({
            "trained_at": pd.Timestamp.now().isoformat(),
            "metrics": {"r2": r2, "mae": mae},
            "importances": {k: float(v) for k, v in feat_imp.items()},
            "n_samples": len(X)
        }, f)
    
    print("🚀 Neural Engine Sync Complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", help="Path to dataset")
    args = parser.parse_args()
    train_advanced(args.data)
