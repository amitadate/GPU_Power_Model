#!/usr/bin/env python3

import joblib
import numpy as np
import pandas as pd
from pathlib import Path
import sys

def optimize_simple_model():
    model_path = Path("models/dvfs_simple_model.pkl")
    
    if not model_path.exists():
        print(f"Simple model file not found at {model_path}")
        return False
    
    try:
        # Load the model
        model_info = joblib.load(model_path)
        print(f"Loaded model of type: {model_info['model_type']}")
        
        # Get the model
        model = model_info['model']
        
        # Print current coefficients
        print("Current coefficients:")
        print(f"  Coefficients: {model.coef_}")
        print(f"  Intercept: {model.intercept_}")
        
        # weighting to optimize coefficients
        model.coef_[0] = 0.6    # Increase utilization weight slightly
        model.coef_[1] = 0.022  # Slightly decrease clock weight
        model.intercept_ = 12   # Increase base power slightly
        
        # Save the updated model
        joblib.dump(model_info, model_path)
        print(f"Updated model saved to {model_path}")
        print(f"New coefficients:")
        print(f"  Coefficients: {model.coef_}")
        print(f"  Intercept: {model.intercept_}")
        
        return True
            
    except Exception as e:
        print(f"Error optimizing model: {e}")
        return False

if __name__ == "__main__":
    print("DVFS Model Optimization Tool")
    optimize_simple_model()
    print("Done")
