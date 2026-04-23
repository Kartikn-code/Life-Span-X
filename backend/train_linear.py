import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import json
import os

def generate_health_data(n=10000):
    np.random.seed(42)
    
    # Features
    age = np.random.randint(18, 90, n)
    gender = np.random.choice([0, 1], n) # 0: Female, 1: Male
    bmi = np.random.uniform(15, 45, n)
    exercise = np.random.randint(0, 4, n) # 0-3 level
    smoking = np.random.choice([0, 1], n, p=[0.7, 0.3])
    alcohol = np.random.choice([0, 1], n, p=[0.6, 0.4])
    systolic = np.random.randint(90, 180, n)
    cholesterol = np.random.randint(140, 300, n)
    glucose = np.random.randint(70, 200, n)
    
    # Target (Lifespan) - Real-world inspired weights
    # Base 82 for female, 79 for male
    y = np.where(gender == 0, 82.0, 79.0).astype(float)
    
    # Learned impacts (simulated ground truth)
    y += (40 - age) * 0.1  # Age impact
    y -= smoking * 12
    y -= alcohol * 5
    y += exercise * 3.5
    y -= np.where(bmi > 25, (bmi - 25) * 0.6, 0)
    y -= np.where(systolic > 140, 6, 0)
    y -= np.where(cholesterol > 240, 4, 0)
    y -= np.where(glucose > 120, 5, 0)
    
    # Add noise
    y += np.random.normal(0, 2, n)
    
    # Clamp
    y = np.clip(y, age + 1, 100)
    
    return pd.DataFrame({
        'age': age, 'gender': gender, 'bmi': bmi, 'exercise': exercise,
        'smoking': smoking, 'alcohol': alcohol, 'systolic': systolic,
        'cholesterol': cholesterol, 'glucose': glucose
    }), y

def train_linear():
    print("Generating training data...")
    X, y = generate_health_data()
    
    print("Training Linear Regression...")
    model = LinearRegression()
    model.fit(X, y)
    
    print("\n--- MODEL PARAMETERS ---")
    print(f"Intercept: {model.intercept_}")
    
    coef_map = {}
    for name, coef in zip(X.columns, model.coef_):
        print(f"{name}: {coef}")
        coef_map[name] = float(coef)
    
    # Export for JS
    results = {
        "intercept": float(model.intercept_),
        "coefficients": coef_map,
        "features": list(X.columns),
        "r2_score": float(model.score(X, y))
    }
    
    with open(os.path.join(os.path.dirname(__file__), 'linear_weights.json'), 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\nWeights saved to linear_weights.json")

if __name__ == "__main__":
    train_linear()
