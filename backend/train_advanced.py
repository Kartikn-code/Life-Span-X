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
NUMERIC_FEATURES = ['age', 'bmi', 'blood_pressure', 'cholesterol', 'glucose', 'sleep_hours']
BINARY_FEATURES = ['gender', 'smoking', 'alcohol', 'heart_disease', 'diabetes', 'stroke']
ORDINAL_FEATURES = ['exercise_level', 'stress_level']
ALL_FEATURES = NUMERIC_FEATURES + BINARY_FEATURES + ORDINAL_FEATURES

# Supabase Setup (Handles both Backend and Vite-prefixed env vars)
SUPABASE_URL = os.environ.get('SUPABASE_URL') or os.environ.get('VITE_SUPABASE_URL')
SUPABASE_KEY = os.environ.get('SUPABASE_KEY') or os.environ.get('VITE_SUPABASE_ANON_KEY')

def get_supabase():
    if SUPABASE_URL and SUPABASE_KEY:
        try:
            return create_client(SUPABASE_URL, SUPABASE_KEY)
        except Exception as e:
            print(f"Cloud Connection Failed: {e}")
    return None

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
    if 'age' not in df.columns:
        df['age'] = np.random.randint(25, 80, size=len(df))
    for col in df.columns:
        if col in ALL_FEATURES or col == 'lifespan':
            df[col] = pd.to_numeric(df[col], errors='coerce')
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
    supabase = get_supabase()
    data_dir = os.path.join(os.path.dirname(__file__), 'data')
    os.makedirs(data_dir, exist_ok=True)
    master_path = os.path.join(data_dir, 'master_training_data.csv')
    
    if data_path:
        print(f"Processing new data from {data_path}...")
        try:
            df_new = pd.read_excel(data_path, engine='openpyxl') if data_path.endswith('.xlsx') else pd.read_csv(data_path)
            df_new = map_headers(df_new)
            X_new = df_new[ALL_FEATURES]
            y_new = df_new['lifespan'].fillna(78.0) if 'lifespan' in df_new.columns else 82.0 - (X_new['bmi'] - 22.5).clip(0)*0.4 - X_new['smoking']*10 + X_new['exercise_level']*2.5
            
            df_clean = X_new.copy()
            df_clean['lifespan'] = y_new
            df_clean = df_clean.dropna(subset=['lifespan'])
            
            # Sync to Cloud (Supabase) - Wrap in try/except for graceful fallback
            if supabase:
                try:
                    print("☁️ Syncing data to Supabase Cloud Storage...")
                    records = df_clean.to_dict('records')
                    for i in range(0, len(records), 1000):
                        supabase.table('training_data').insert(records[i:i+1000]).execute()
                    print("✅ Cloud sync successful.")
                except Exception as cloud_err:
                    print(f"⚠️ Cloud Sync failed (Table might not exist): {cloud_err}")
                    print("💾 Falling back to Local Master Database.")
            
            # Update Local Master
            if os.path.exists(master_path):
                df_master = pd.read_csv(master_path)
                df_combined = pd.concat([df_master, df_clean], ignore_index=True).drop_duplicates()
                df_combined.to_csv(master_path, index=False)
            else:
                df_clean.to_csv(master_path, index=False)
                df_combined = df_clean
        except Exception as e:
            print(f"❌ Data Sync Error: {e}")
            return
    
    # Load for Training: Try Cloud first, then Local
    df_train = None
    if supabase:
        try:
            print("🌐 Fetching latest clinical records from Cloud...")
            response = supabase.table('training_data').select("*").execute()
            if response.data:
                df_train = pd.DataFrame(response.data).drop(columns=['id', 'created_at'], errors='ignore')
                print(f"✅ Loaded {len(df_train)} cloud records.")
        except Exception as e:
            print(f"⚠️ Cloud fetch failed, falling back to local: {e}")

    if df_train is None:
        if os.path.exists(master_path):
            print("💾 Loading from Local Master Database...")
            df_train = pd.read_csv(master_path)
        else:
            print("🎲 No Cloud or Local records found. Generating 2,000 records to bootstrap the system...")
            X, y = generate_realistic_data(2000)
            df_train = X.assign(lifespan=y)
            # Bootstrapping: Save this initial intelligence to the cloud - Wrap in try/except
            if supabase:
                try:
                    print("📤 Bootstrapping Supabase Cloud with initial records...")
                    records = df_train.to_dict('records')
                    for i in range(0, len(records), 1000):
                        supabase.table('training_data').insert(records[i:i+1000]).execute()
                except Exception as boot_err:
                    print(f"⚠️ Cloud Bootstrap failed (Table might not exist): {boot_err}")

    X = df_train[ALL_FEATURES]
    y = df_train['lifespan']
    
    # Render Free Tier Protection: If dataset is too large, use a smart sample of 5,000
    if len(X) > 5000:
        print(f"📉 Large dataset detected ({len(X)} rows). Optimizing for Cloud RAM (using 5,000 sample records)...")
        indices = np.random.choice(len(X), 5000, replace=False)
        X = X.iloc[indices]
        y = y.iloc[indices]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    preprocessor = ColumnTransformer(transformers=[
        ('num', StandardScaler(), NUMERIC_FEATURES),
        ('pass', 'passthrough', BINARY_FEATURES + ORDINAL_FEATURES)
    ])
    # Optimized for Cloud (Render Free Tier) Memory Limits
    model = RandomForestRegressor(
        n_estimators=100, 
        max_depth=10, 
        min_samples_leaf=5, 
        random_state=42,
        n_jobs=1 # Use single core to save RAM
    )
    pipeline = Pipeline([('preprocessor', preprocessor), ('model', model)])
    
    print(f"🧠 Training Brain on {len(X)} clinical records (Memory Optimized)...")
    pipeline.fit(X_train, y_train)
    
    mae = mean_absolute_error(y_test, pipeline.predict(X_test))
    r2 = r2_score(y_test, pipeline.predict(X_test))
    
    print(f"\n--- AI Brain Summary ---\nMemory Capacity: {len(X)} Records\nIntelligence (R²): {r2:.4f}\nError Margin (MAE): {mae:.2f} years")
    
    feat_imp = sorted(zip(ALL_FEATURES, pipeline.named_steps['model'].feature_importances_), key=lambda x: x[1], reverse=True)
    backend_dir = os.path.dirname(__file__)
    joblib.dump(pipeline, os.path.join(backend_dir, 'lifespan_advanced.joblib'))
    
    with open(os.path.join(backend_dir, 'features.json'), 'w') as f:
        json.dump({
            "features": ALL_FEATURES, "metrics": {"r2": r2, "mae": mae},
            "n_samples": len(X), "trained_at": pd.Timestamp.now().isoformat(),
            "importances": {name: float(imp) for name, imp in feat_imp}
        }, f, indent=2)
    print("✨ Model globally synced and saved.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", help="Path to CSV or XLSX dataset")
    args = parser.parse_args()
    train_advanced(args.data)
