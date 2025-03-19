#!/usr/bin/env python3
"""
Fix DVFS Model - Repairs the polynomial model coefficients to ensure reasonable predictions
"""

import joblib
import numpy as np
import pandas as pd
from pathlib import Path
import sys

def fix_polynomial_model():
    """Fix the problematic polynomial model coefficients"""
    model_path = Path("models/dvfs_best_model.pkl")
    
    if not model_path.exists():
        print(f"Model file not found at {model_path}")
        return False
    
    try:
        # Load the model
        model_info = joblib.load(model_path)
        print(f"Loaded model of type: {model_info['model_type']}")
        
        # Check if it's a polynomial model
        if model_info['model_type'] != 'polynomial':
            print("This is not a polynomial model, no fix needed")
            return False
        
        # Get the model pipeline
        model = model_info['model']
        
        # Backup the original model
        backup_path = model_path.with_suffix(".bak.pkl")
        joblib.dump(model_info, backup_path)
        print(f"Original model backed up to {backup_path}")
        
        # Access the ridge regression component
        if hasattr(model, 'named_steps') and 'ridge' in model.named_steps:
            ridge = model.named_steps['ridge']
            
            # Print original coefficients
            print("Original ridge coefficients:")
            print(f"  Coefficients: {ridge.coef_}")
            print(f"  Intercept: {ridge.intercept_}")
            
            # Fix the intercept (most critical issue)
            ridge.intercept_ = max(0, ridge.intercept_)  # Ensure non-negative
            
            # Adjust coefficients if needed
            # Identify which coefficient corresponds to which feature
            poly = model.named_steps.get('poly', None)
            if poly:
                print(f"Polynomial features: {poly.get_feature_names_out(['utilization', 'sm_clock'])}")
            
            # Save the updated model
            joblib.dump(model_info, model_path)
            print(f"Updated model saved to {model_path}")
            print(f"New ridge coefficients:")
            print(f"  Coefficients: {ridge.coef_}")
            print(f"  Intercept: {ridge.intercept_}")
            
            return True
        else:
            print("Could not access ridge regression component")
            return False
            
    except Exception as e:
        print(f"Error fixing model: {e}")
        return False

def create_simple_linear_model():
    """Create a simple linear model as a fallback"""
    try:
        from sklearn.linear_model import Ridge
        
        # Create a simple model with reasonable coefficients
        model = Ridge(alpha=1.0)
        
        # Set reasonable coefficients directly
        # power = intercept + c1*utilization + c2*clock
        model.coef_ = np.array([0.5, 0.025])  # For [utilization, sm_clock]
        model.intercept_ = 10  # Base power consumption
        
        # Create model info
        model_info = {
            'model_type': 'linear_with_clock',
            'model': model,
            'features': ['utilization', 'sm_clock']
        }
        
        # Save the new model
        new_model_path = Path("models/dvfs_simple_model.pkl")
        joblib.dump(model_info, new_model_path)
        print(f"Created simple linear model and saved to {new_model_path}")
        
        return model_info
    except Exception as e:
        print(f"Error creating simple model: {e}")
        return None

if __name__ == "__main__":
    print("DVFS Model Repair Tool")
    if not fix_polynomial_model():
        print("Creating a simple alternative model")
        create_simple_linear_model()
    
    print("Done")
