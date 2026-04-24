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
import time

# Feature Names and Types
NUMERIC_FEATURES = ['age', 'bmi', 'blood_pressure', 'cholesterol', 'glucose', 'sleep_hours']
BINARY_FEATURES = ['gender', 'smoking', 'alcohol', 'heart_disease', 'diabetes', 'stroke']
ORDINAL_FEATURES = ['exercise_level', 'stress_level']
ALL_FEATURES = NUMERIC_FEATURES + BINARY_FEATURES + ORDINAL_FEATURES

def generate_realistic_data(n=10000):
    np.random.seed(42)
    age = np.clip(np.random.normal(52, 15, n), 18, 90)
    gender = np.random.choice([0, 1], n, p=[0.51, 0.49])
    bmi = np.clip(np.random.lognormal(mean=3.35, sigma=0.15, size=n), 15, 55)
    exercise_level = np.random.randint(0, 4, n)
    smoking = np.random.choice([0, 1], n, p=[0.78, 0.22])
    alcohol = np.random.choice([0, 1], n, p=[0.65, 0.35])
    blood_pressure = np.random.normal(125, 15, n) + (age-40)*0.3
    cholesterol = np.random.normal(190, 40, n) + (bmi-25)*2
    glucose = np.random.normal(95, 25, n) + (bmi-25)*1.5
    
    p_diabetes = np.clip(0.05 + (bmi-25)*0.01 + (age-40)*0.002, 0.01, 0.4)
    diabetes = np.array([np.random.choice([0, 1], p=[1-p, p]) for p in p_diabetes])
    
    p_heart = np.clip(0.03 + (age-50)*0.005 + smoking*0.05, 0.01, 0.3)
    heart_disease = np.array([np.random.choice([0, 1], p=[1-p, p]) for p in p_heart])
    
    stroke = np.random.choice([0, 1], n, p=[0.97, 0.03])
    sleep_hours = np.random.normal(7, 1, n)
    stress_level = np.random.randint(1, 6, n)
    
    y = np.where(gender == 0, 82.0, 77.8)
    y -= (bmi - 22.5).clip(0) * 0.4
    y -= smoking * 10.0
    y -= (blood_pressure - 120).clip(0) * 0.15
    y -= diabetes * 6.0
    y -= heart_disease * 7.0
    y += exercise_level * 2.5
    y += (sleep_hours - 7) * 0.5
    y -= (stress_level - 3) * 1.2
    y += np.random.normal(0, 3, n)
    y = np.clip(y, age + 2, 105)
    
    X = pd.DataFrame({
        'age': age, 'gender': gender, 'bmi': bmi, 'exercise_level': exercise_level,
        'smoking': smoking, 'alcohol': alcohol, 'blood_pressure': blood_pressure,
        'cholesterol': cholesterol, 'glucose': glucose, 'heart_disease': heart_disease,
        'diabetes': diabetes, 'stroke': stroke, 'sleep_hours': sleep_hours, 'stress_level': stress_level
    })
    return X, y

def map_headers(df):
    """Deep synonym mapping for clinical data standard compatibility"""
    # Standardize column names (remove extra spaces, lowercase)
    df.columns = [c.strip() for c in df.columns]
    
    mapping = {
        'lifespan': ['Life expectancy', 'Expected Lifespan', 'Target', 'years', 'longevity'],
        'age': ['Age', 'Patient Age', 'Individual Age'],
        'bmi': ['BMI', 'Body Mass Index', 'bmi'],
        'smoking': ['Smoking', 'Smoker', 'smoking_status', 'Smoking Level', 'Smokes'],
        'alcohol': ['Alcohol', 'Drinking', 'alcohol_consumption', 'Drinks'],
        'gender': ['Gender', 'Sex', 'sex'],
        'blood_pressure': ['blood_pressure', 'bp', 'systolic', 'Blood Pressure', 'SBP'],
        'cholesterol': ['Cholesterol', 'chol', 'Total Cholesterol', 'TC'],
        'glucose': ['Glucose', 'glucose_level', 'Sugar', 'Blood Sugar'],
        'heart_disease': ['Heart Disease', 'heart_disease', 'cvd', 'Heart Condition'],
        'diabetes': ['Diabetes', 'diabetes_status', 'Diabetic'],
        'stroke': ['Stroke', 'stroke_status'],
        'exercise_level': ['Exercise', 'Physical Activity', 'exercise', 'Activity Level'],
        'sleep_hours': ['Sleep', 'sleep', 'hours_of_sleep', 'Sleep Duration'],
        'stress_level': ['Stress', 'stress', 'mental_health', 'Stress Level']
    }
    
    new_cols = {}
    for canonical, synonyms in mapping.items():
        for syn in synonyms:
            if syn in df.columns:
                new_cols[syn] = canonical
                break
    
    df = df.rename(columns=new_cols)
    
    # If age is missing but Year is present, WHO datasets are aggregated, 
    # so we assign a realistic median age distribution for training stability.
    if 'age' not in df.columns:
        df['age'] = np.random.randint(25, 80, size=len(df))
        
    # Force features to numeric
    for col in df.columns:
        if col in ALL_FEATURES or col == 'lifespan':
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Fill defaults
    for f in ALL_FEATURES:
        if f not in df.columns:
            df[f] = 0
        if f == 'age': df[f] = df[f].fillna(45)
        elif f == 'bmi': df[f] = df[f].fillna(24.5)
        elif f == 'sleep_hours': df[f] = df[f].fillna(7)
        elif f in ['stress_level', 'exercise_level']: df[f] = df[f].fillna(2)
        else: df[f] = df[f].fillna(0)
            
    return df

def train_advanced(data_path=None):
    data_dir = os.path.join(os.path.dirname(__file__), 'data')
    os.makedirs(data_dir, exist_ok=True)
    master_path = os.path.join(data_dir, 'master_training_data.csv')
    
    if data_path:
        print(f"Loading and Merging new data from {data_path}...")
        try:
            if data_path.endswith('.xlsx'):
                df_new = pd.read_excel(data_path, engine='openpyxl')
            else:
                df_new = pd.read_csv(data_path)
            
            df_new = map_headers(df_new)
            
            # Prepare for merging
            X_new = df_new[ALL_FEATURES]
            if 'lifespan' in df_new.columns and not df_new['lifespan'].isnull().all():
                y_new = df_new['lifespan'].fillna(78.0)
            else:
                print("No target 'lifespan' found. Generating via health metrics...")
                y_new = 82.0 - (X_new['bmi'] - 22.5).clip(0)*0.4 - X_new['smoking']*10 + X_new['exercise_level']*2.5
            
            df_clean = X_new.copy()
            df_clean['lifespan'] = y_new
            df_clean = df_clean.dropna(subset=['lifespan'])
            
            # Cumulative Merge
            if os.path.exists(master_path):
                df_master = pd.read_csv(master_path)
                df_combined = pd.concat([df_master, df_clean], ignore_index=True)
                df_combined = df_combined.drop_duplicates()
                df_combined.to_csv(master_path, index=False)
                print(f"Memory Extended: Database now has {len(df_combined)} unique clinical records.")
            else:
                df_clean.to_csv(master_path, index=False)
                df_combined = df_clean
                print(f"Initialized Memory: {len(df_combined)} records saved.")
                
            X = df_combined[ALL_FEATURES]
            y = df_combined['lifespan']
        except Exception as e:
            print(f"Data Processing Error: {e}")
            return
    else:
        if os.path.exists(master_path):
            print(f"Training on Master database records...")
            df_master = pd.read_csv(master_path)
            X = df_master[ALL_FEATURES]
            y = df_master['lifespan']
        else:
            print("No data found. Generating 10,000 synthetic samples...")
            X, y = generate_realistic_data(10000)
    
    # Save a copy for inspection
    X.assign(lifespan=y).to_csv(os.path.join(data_dir, 'last_training_data.csv'), index=False)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    preprocessor = ColumnTransformer(transformers=[
        ('num', StandardScaler(), NUMERIC_FEATURES),
        ('pass', 'passthrough', BINARY_FEATURES + ORDINAL_FEATURES)
    ])
    
    # Increased estimators for better accuracy with combined data
    model = RandomForestRegressor(n_estimators=300, max_depth=15, min_samples_leaf=3, random_state=42)
    pipeline = Pipeline([('preprocessor', preprocessor), ('model', model)])
    
    print(f"Retraining Brain on {len(X)} records...")
    pipeline.fit(X_train, y_train)
    
    mae = mean_absolute_error(y_test, pipeline.predict(X_test))
    r2 = r2_score(y_test, pipeline.predict(X_test))
    
    print(f"\n--- Cumulative Retraining Summary ---\nTotal Data Points: {len(X)}\nR² Score: {r2:.4f}\nMAE: {mae:.2f} years")
    
    feat_imp = sorted(zip(ALL_FEATURES, pipeline.named_steps['model'].feature_importances_), key=lambda x: x[1], reverse=True)
    backend_dir = os.path.dirname(__file__)
    joblib.dump(pipeline, os.path.join(backend_dir, 'lifespan_advanced.joblib'))
    
    with open(os.path.join(backend_dir, 'features.json'), 'w') as f:
        json.dump({
            "features": ALL_FEATURES, "metrics": {"r2": r2, "mae": mae},
            "n_samples": len(X), "trained_at": pd.Timestamp.now().isoformat(),
            "importances": {name: float(imp) for name, imp in feat_imp}
        }, f, indent=2)
    print(f"Model successfully saved to {backend_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", help="Path to CSV or XLSX dataset")
    args = parser.parse_args()
    train_advanced(args.data)
