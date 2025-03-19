#!/usr/bin/env python3


import joblib
import numpy as np
import pandas as pd
from pathlib import Path
import sys
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error

def load_enhanced_power_data():
    # Find the most recent enhanced data file
    power_results = Path("power_results")
    enhanced_files = list(power_results.glob("enhanced_histogram_power_data_*.csv"))
    
    if not enhanced_files:
        print("No enhanced power data file found.")
        sys.exit(1)
    
    most_recent_file = max(enhanced_files, key=lambda p: p.stat().st_mtime)
    print(f"Loading data from {most_recent_file}")
    
    # Load the enhanced data
    df = pd.read_csv(most_recent_file)
    
    # Filter out rows with NaN values
    df = df.dropna(subset=['power_watts', 'utilization', 'sm_clock', 'memory_bandwidth_pct'])
    
    # Filter out transition states (where clock changes significantly)
    df = df[(df['sm_clock'].diff().abs() < 100) | (df['sm_clock'].diff().isna())]
    
    return df

def create_memory_aware_cubic_model(data):
    # Normalize data for better modeling
    max_sm_clock = 2000  # Typical max for RTX series
    
    # Create derived features
    df = data.copy()
    df['clock_ghz'] = df['sm_clock'] / 1000.0  # Convert to GHz
    df['f_cubed'] = df['clock_ghz'] ** 3
    df['f_linear'] = df['clock_ghz']
    df['mem_factor'] = 1.0 + (df['memory_bandwidth_pct'] / 100.0) * 0.5
    df['f_cubed_mem'] = df['f_cubed'] * df['mem_factor']
    
    # Create feature matrix
    X = df[['utilization', 'f_cubed_mem', 'f_linear']].values
    y = df['power_watts'].values
    
    # Create and fit the model
    # Ridge regression with regularization to prevent overfitting
    model = Ridge(alpha=0.1)
    model.fit(X, y)
    
    # Evaluate model on training data
    y_pred = model.predict(X)
    mae = mean_absolute_error(y, y_pred)
    rmse = np.sqrt(mean_squared_error(y, y_pred))
    
    print(f"Model training results:")
    print(f"  Mean Absolute Error: {mae:.2f} W")
    print(f"  Root Mean Squared Error: {rmse:.2f} W")
    print(f"  Coefficients: {model.coef_}")
    print(f"  Intercept: {model.intercept_}")
    
    # Package the model information
    model_info = {
        'model': model,
        'model_type': 'memory_aware_cubic_dvfs',
        'features': ['utilization', 'f_cubed_mem', 'f_linear'],
        'stats': {
            'mae': mae,
            'rmse': rmse
        }
    }
    
    return model_info

def create_full_feature_model(data):
    # Normalize data for better modeling
    max_sm_clock = 2000  # Typical max for RTX series
    
    # Create derived features
    df = data.copy()
    df['clock_ghz'] = df['sm_clock'] / 1000.0  # Convert to GHz
    
    # Create feature matrix - using original features
    X = df[['utilization', 'clock_ghz', 'memory_bandwidth_pct']].values
    y = df['power_watts'].values
    
    # Create a polynomial features pipeline with Ridge regression
    model = Pipeline([
        ('poly', PolynomialFeatures(degree=2, include_bias=True)),
        ('ridge', Ridge(alpha=0.1))
    ])
    
    model.fit(X, y)
    
    # Evaluate model on training data
    y_pred = model.predict(X)
    mae = mean_absolute_error(y, y_pred)
    rmse = np.sqrt(mean_squared_error(y, y_pred))
    
    print(f"Full feature model training results:")
    print(f"  Mean Absolute Error: {mae:.2f} W")
    print(f"  Root Mean Squared Error: {rmse:.2f} W")
    
    # Get coefficients from Ridge regression
    ridge_coef = model.named_steps['ridge'].coef_
    print(f"  Number of polynomial features: {len(ridge_coef)}")
    print(f"  Ridge intercept: {model.named_steps['ridge'].intercept_}")
    
    # Package the model information
    model_info = {
        'model': model,
        'model_type': 'memory_aware_polynomial',
        'features': ['utilization', 'clock_ghz', 'memory_bandwidth_pct'],
        'stats': {
            'mae': mae,
            'rmse': rmse
        }
    }
    
    return model_info

def main():
    # Load power data with memory metrics
    power_data = load_enhanced_power_data()
    
    # Create models directory if it doesn't exist
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    
    # Create and save the memory-aware cubic model
    cubic_model_info = create_memory_aware_cubic_model(power_data)
    cubic_model_path = models_dir / "memory_aware_cubic_model.pkl"
    joblib.dump(cubic_model_info, cubic_model_path)
    print(f"Memory-aware cubic DVFS model saved to {cubic_model_path}")
    
    # Create and save the full feature model
    full_model_info = create_full_feature_model(power_data)
    full_model_path = models_dir / "memory_aware_full_model.pkl"
    joblib.dump(full_model_info, full_model_path)
    print(f"Memory-aware full feature model saved to {full_model_path}")
    
    # Compare with existing models
    try:
        # Load basic model
        basic_model_path = models_dir / "basic_power_model.pkl"
        basic_model = joblib.load(basic_model_path)
        
        # Load DVFS cubic model
        cubic_dvfs_model_path = models_dir / "dvfs_cubic_dvfs_model.pkl"
        cubic_dvfs_model = joblib.load(cubic_dvfs_model_path)
        
        print("\nModel comparison:")
        print("  Basic model and DVFS cubic model loaded for reference")
    except Exception as e:
        print(f"Could not load existing models for comparison: {e}")

if __name__ == "__main__":
    main()
