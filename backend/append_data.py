import pandas as pd
from train_advanced import generate_realistic_data, ALL_FEATURES
import os

def append_data():
    X, y = generate_realistic_data(600)
    df = X.copy()
    df['lifespan'] = y
    
    path = 'data/sample_health_data.csv'
    if os.path.exists(path):
        existing = pd.read_csv(path)
        combined = pd.concat([existing, df], ignore_index=True)
        combined.to_csv(path, index=False)
        print(f"Appended 600 rows to {path}. Total: {len(combined)}")
    else:
        df.to_csv(path, index=False)
        print(f"Created {path} with 600 rows.")

if __name__ == "__main__":
    append_data()
