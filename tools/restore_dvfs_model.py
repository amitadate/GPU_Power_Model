#!/usr/bin/env python3
"""
Restore DVFS Model - Reverts the simple linear model coefficients to the original values
that performed better (53.45% improvement)
"""

import joblib
import numpy as np
import pandas as pd
from pathlib import Path
import sys

def restore_simple_model():
    """Restore the simple linear model to the original, better coefficients"""
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
        
        # Restore the original coefficients that performed better
        model.coef_[0] = 0.5     # Original utilization weight
        model.coef_[1] = 0.025   # Original clock weight
        model.intercept_ = 10    # Original base power
        
        # Save the updated model
        joblib.dump(model_info, model_path)
        print(f"Updated model saved to {model_path}")
        print(f"Restored coefficients:")
        print(f"  Coefficients: {model.coef_}")
        print(f"  Intercept: {model.intercept_}")
        
        return True
            
    except Exception as e:
        print(f"Error restoring model: {e}")
        return False

if __name__ == "__main__":
    print("DVFS Model Restoration Tool")
    restore_simple_model()
    print("Done")
