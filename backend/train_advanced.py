import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
import joblib
import os

def generate_complex_data(n=10000):
    np.random.seed(42)
    
    # Features
    age = np.random.randint(18, 90, n)
    gender = np.random.choice([0, 1], n)
    bmi = np.random.uniform(15, 45, n)
    exercise = np.random.randint(0, 4, n)
    smoking = np.random.choice([0, 1], n, p=[0.7, 0.3])
    alcohol = np.random.choice([0, 1], n, p=[0.6, 0.4])
    systolic = np.random.randint(90, 180, n)
    cholesterol = np.random.randint(140, 300, n)
    glucose = np.random.randint(70, 200, n)
    
    # Target with non-linear interactions
    y = 80.0
    
    # Age-BMI interaction (BMI is more dangerous as you age)
    y -= (age/20) * np.where(bmi > 25, (bmi-25)*0.4, 0)
    
    # Smoking-BP synergy
    y -= smoking * 10
    y -= np.where((smoking == 1) & (systolic > 140), 5, 0)
    
    # Exercise benefits
    y += exercise * 3.0
    
    # Random noise
    y += np.random.normal(0, 3, n)
    y = np.clip(y, age + 2, 100)
    
    X = pd.DataFrame({
        'age': age, 'gender': gender, 'bmi': bmi, 'exercise': exercise,
        'smoking': smoking, 'alcohol': alcohol, 'systolic': systolic,
        'cholesterol': cholesterol, 'glucose': glucose
    })
    
    return X, y

def train_advanced():
    print("Generating complex training data...")
    X, y = generate_complex_data()
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print("Training Random Forest Regressor...")
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('model', RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42))
    ])
    
    pipeline.fit(X_train, y_train)
    
    score = pipeline.score(X_test, y_test)
    print(f"Model R2 Score: {score:.4f}")
    
    # Save the pipeline
    model_path = os.path.join(os.path.dirname(__file__), 'lifespan_advanced.joblib')
    joblib.dump(pipeline, model_path)
    print(f"Advanced model saved to {model_path}")
    
    # Save feature names for the API
    with open(os.path.join(os.path.dirname(__file__), 'features.json'), 'w') as f:
        import json
        json.dump(list(X.columns), f)

if __name__ == "__main__":
    train_advanced()
