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
ORDINAL_FEATURES = []
ALL_FEATURES = NUMERIC_FEATURES + BINARY_FEATURES + ORDINAL_FEATURES

# Supabase Setup
SUPABASE_URL = os.environ.get('SUPABASE_URL') or os.environ.get('VITE_SUPABASE_URL')
SUPABASE_KEY = os.environ.get('SUPABASE_KEY') or os.environ.get('VITE_SUPABASE_ANON_KEY')

def get_supabase():
    if SUPABASE_URL and SUPABASE_KEY:
        try: return create_client(SUPABASE_URL, SUPABASE_KEY)
        except: return None
    return None

def generate_realistic_data(n=10000):
    np.random.seed(42)
    age = np.clip(np.random.normal(52, 15, n), 18, 90)
    gender = np.random.choice([0, 1], n)
    bmi = np.clip(np.random.normal(26, 6, n), 15, 50)
    
    # Lifestyle (Continuous 0.0-1.0)
    exercise_level = np.random.uniform(0, 1, n)
    smoking = np.random.uniform(0, 1, n)
    alcohol = np.random.uniform(0, 1, n)
    stress_level = np.random.uniform(0, 1, n)
    
    # Clinical (Realistic Ranges)
    blood_pressure = np.random.normal(128, 18, n) + (bmi-25)*0.6
    cholesterol = np.random.normal(205, 45, n) + (bmi-25)*1.5
    glucose = np.random.normal(100, 30, n) + (bmi-25)*1.0
    sleep_hours = np.random.normal(6.8, 1.2, n)
    
    # Conditions
    diabetes = np.random.choice([0, 1], n, p=[0.88, 0.12])
    heart_disease = np.random.choice([0, 1], n, p=[0.93, 0.07])
    stroke = np.random.choice([0, 1], n, p=[0.98, 0.02])
    
    # 🎯 EXTREME SENSITIVITY ENGINE
    y = 80.0 
    y -= (smoking * 20.0)
    y += (exercise_level * 15.0)
    y -= (alcohol * 12.0)
    y -= (stress_level * 10.0)
    y += (sleep_hours - 7) * 2.5
    y -= (diabetes * 15.0)
    y -= (heart_disease * 18.0)
    y -= (stroke * 20.0)
    y -= ((blood_pressure - 120).clip(0) * 0.4)
    y -= ((cholesterol - 200).clip(0) * 0.1)
    y -= ((glucose - 100).clip(0) * 0.15)
    y -= (np.abs(bmi - 22) * 2.5)
    y += (1 - gender) * 5.0
    y += np.random.normal(0, 0.2, n)
    y = np.clip(y, age + 1, 105)
    
    X = pd.DataFrame({
        'age': age, 'gender': gender, 'bmi': bmi, 'exercise_level': exercise_level,
        'smoking': smoking, 'alcohol': alcohol, 'blood_pressure': blood_pressure,
        'cholesterol': cholesterol, 'glucose': glucose, 'heart_disease': heart_disease,
        'diabetes': diabetes, 'stroke': stroke, 'sleep_hours': sleep_hours, 'stress_level': stress_level
    })
    return X, y

def map_headers(df):
    df.columns = [c.strip() for c in df.columns]
    mapping = {
        'lifespan': ['Life expectancy', 'Expected Lifespan', 'Target', 'longevity'],
        'age': ['Age', 'Patient Age'],
        'bmi': ['BMI', 'Body Mass Index'],
        'smoking': ['Smoking', 'Smoker', 'smoking_status'],
        'alcohol': ['Alcohol', 'Drinking', 'alcohol_consumption'],
        'gender': ['Gender', 'Sex'],
        'blood_pressure': ['bp', 'systolic', 'Blood Pressure', 'SBP'],
        'cholesterol': ['Cholesterol', 'chol', 'Total Cholesterol'],
        'glucose': ['Glucose', 'glucose_level', 'Sugar'],
        'heart_disease': ['Heart Disease', 'heart_disease', 'cvd'],
        'diabetes': ['Diabetes', 'diabetes_status'],
        'stroke': ['Stroke', 'stroke_status'],
        'exercise_level': ['Exercise', 'Activity Level'],
        'sleep_hours': ['Sleep', 'Sleep Duration'],
        'stress_level': ['Stress', 'Stress Level']
    }
    new_cols = {}
    for canonical, synonyms in mapping.items():
        for syn in synonyms:
            if syn in df.columns:
                new_cols[syn] = canonical
                break
    df = df.rename(columns=new_cols)
    for col in df.columns:
        if col in ALL_FEATURES or col == 'lifespan':
            df[col] = pd.to_numeric(df[col], errors='coerce')
    for f in ALL_FEATURES:
        if f not in df.columns:
            df[f] = 0
        if f == 'age': df[f] = df[f].fillna(45)
        elif f == 'bmi': df[f] = df[f].fillna(24.5)
        elif f == 'sleep_hours': df[f] = df[f].fillna(7)
        else: df[f] = df[f].fillna(0)
    return df

def train_advanced(data_path=None):
    backend_dir = os.path.dirname(__file__)
    data_dir = os.path.join(backend_dir, 'data')
    os.makedirs(data_dir, exist_ok=True)
    master_path = os.path.join(data_dir, 'master_training_data.csv')
    
    X_gen, y_gen = generate_realistic_data(10000)
    df_gen = X_gen.assign(lifespan=y_gen)
    
    if data_path and os.path.exists(data_path):
        print("📂 Blending uploaded data...")
        df_uploaded = pd.read_csv(data_path) if data_path.endswith('.csv') else pd.read_excel(data_path)
        df_uploaded = map_headers(df_uploaded)
        df_train = pd.concat([df_uploaded, df_gen], ignore_index=True).dropna(subset=['lifespan'])
    else:
        print("🎲 Using High-Sensitivity medical baseline...")
        df_train = df_gen
    
    df_train.to_csv(master_path, index=False)
    X = df_train[ALL_FEATURES]
    y = df_train['lifespan']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    preprocessor = ColumnTransformer(transformers=[
        ('num', StandardScaler(), NUMERIC_FEATURES),
        ('pass', 'passthrough', BINARY_FEATURES)
    ])
    pipeline = Pipeline([('preprocessor', preprocessor), ('model', RandomForestRegressor(n_estimators=100, max_depth=12, random_state=42))])
    
    print(f"🧠 Training Brain on {len(X)} records...")
    pipeline.fit(X_train, y_train)
    
    r2 = r2_score(y_test, pipeline.predict(X_test))
    mae = mean_absolute_error(y_test, pipeline.predict(X_test))
    print(f"✨ Intelligence (R²): {r2:.4f}")
    
    joblib.dump(pipeline, os.path.join(backend_dir, 'lifespan_advanced.joblib'))
    ordered_features = NUMERIC_FEATURES + BINARY_FEATURES
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
