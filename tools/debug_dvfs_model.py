#!/usr/bin/env python3
"""
Debug script to investigate why the DVFS model is producing extremely high errors.
This script examines specific data points and traces the prediction calculations.
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import joblib

def find_most_recent_file(directory, pattern):
    """Find the most recent file matching a pattern in a directory."""
    directory = Path(directory)
    files = list(directory.glob(pattern))
    if not files:
        return None
    return max(files, key=lambda p: p.stat().st_mtime)

def load_data():
    """Load the most recent enhanced power data and extract sample points to debug."""
    enhanced_file = find_most_recent_file("power_results", "enhanced_histogram_power_data_*.csv")
    if not enhanced_file:
        print("No enhanced power data found. Please run the power predictor first.")
        sys.exit(1)
    
    print(f"Loading data from {enhanced_file}")
    df = pd.read_csv(enhanced_file)
    
    # Extract some key debugging information
    print(f"Total samples: {len(df)}")
    print(f"Columns: {df.columns.tolist()}")
    
    # Check for expected calculation columns
    expected_cols = ['clock_ratio', 'f_cubed_normalized', 'util_x_clock']
    for col in expected_cols:
        if col not in df.columns:
            print(f"WARNING: Expected column '{col}' not found in data!")
    
    return df

def examine_sample_points(df):
    """Examine specific sample points to understand the prediction issues."""
    print("\n===== EXAMINING SAMPLE POINTS =====")
    
    # Get sample points from each power state
    sample_points = []
    
    # Idle sample (low power, low clock)
    idle_sample = df[(df['sm_clock'] < 400) & (df['power_watts'] < 20)].iloc[0]
    sample_points.append(("Idle", idle_sample))
    
    # Medium sample (if available)
    medium_samples = df[(df['sm_clock'] >= 400) & (df['sm_clock'] < 1000)]
    if not medium_samples.empty:
        medium_sample = medium_samples.iloc[0]
        sample_points.append(("Medium", medium_sample))
    
    # High sample (high power, high clock)
    high_sample = df[(df['sm_clock'] >= 1000) & (df['power_watts'] > 50)].iloc[0]
    sample_points.append(("High", high_sample))
    
    # Examine each sample in detail
    for state_name, sample in sample_points:
        print(f"\n{state_name} State Sample:")
        print(f"  Timestamp: {sample['timestamp']:.1f}")
        print(f"  Actual Power: {sample['power_watts']:.2f} W")
        print(f"  SM Clock: {sample['sm_clock']:.0f} MHz")
        print(f"  Utilization: {sample['utilization']:.1f}%")
        
        # Basic model
        if 'basic_prediction' in sample:
            print(f"  Basic Model Prediction: {sample['basic_prediction']:.2f} W")
            print(f"  Basic Model Error: {sample['basic_error']:.2f} W ({sample['basic_error']/sample['power_watts']*100:.1f}%)")
        
        # DVFS model
        if 'dvfs_prediction' in sample:
            print(f"  DVFS Model Prediction: {sample['dvfs_prediction']:.2f} W")
            print(f"  DVFS Model Error: {sample['dvfs_error']:.2f} W ({sample['dvfs_error']/sample['power_watts']*100:.1f}%)")
        
        # Feature calculations
        if 'clock_ratio' in sample:
            print(f"\n  Calculated Features:")
            print(f"  Clock Ratio: {sample['clock_ratio']:.6f}")
            print(f"  f³ Normalized: {sample['f_cubed_normalized']:.6f}")
            print(f"  Utilization × Clock: {sample['util_x_clock']:.1f}")
            
            # Manually recalculate these values to check
            max_clock = 1950  # This is likely the assumed max clock in MHz
            recalc_clock_ratio = sample['sm_clock'] / max_clock
            recalc_f_cubed = (recalc_clock_ratio ** 3) * 100
            recalc_util_clock = sample['utilization'] * sample['sm_clock']
            
            print(f"\n  Manual Recalculations:")
            print(f"  Max Clock Used: {max_clock} MHz")
            print(f"  Recalculated Clock Ratio: {recalc_clock_ratio:.6f}")
            print(f"  Recalculated f³ Normalized: {recalc_f_cubed:.6f}")
            print(f"  Recalculated Util × Clock: {recalc_util_clock:.1f}")
            
            # Check for discrepancies
            if abs(sample['clock_ratio'] - recalc_clock_ratio) > 0.001:
                print(f"  WARNING: Clock ratio calculation may be incorrect!")
            if abs(sample['f_cubed_normalized'] - recalc_f_cubed) > 0.01:
                print(f"  WARNING: f³ calculation may be incorrect!")
            if abs(sample['util_x_clock'] - recalc_util_clock) > 0.1:
                print(f"  WARNING: Util × Clock calculation may be incorrect!")

def check_model_files():
    """Attempt to locate and examine the model files being used."""
    print("\n===== CHECKING MODEL FILES =====")
    
    # Try to find model file paths from config
    try:
        import config
        print(f"Config imported. Model directories:")
        print(f"  Models directory: {config.MODELS_DIR}")
        print(f"  Basic model filename: {config.MODEL_FILENAME}")
        
        # Check if the model files exist
        basic_model_path = config.MODELS_DIR / config.MODEL_FILENAME
        dvfs_model_path = config.MODELS_DIR / "dvfs_best_model.pkl"
        
        print(f"\nChecking if model files exist:")
        print(f"  Basic model exists: {basic_model_path.exists()}")
        print(f"  DVFS model exists: {dvfs_model_path.exists()}")
        
        # Try to load the models if they exist
        if basic_model_path.exists():
            try:
                basic_model = joblib.load(basic_model_path)
                print(f"\nBasic model loaded successfully")
                if hasattr(basic_model, 'coef_'):
                    print(f"  Basic model coefficients: {basic_model.coef_}")
                    print(f"  Basic model intercept: {basic_model.intercept_}")
            except Exception as e:
                print(f"Error loading basic model: {e}")
        
        if dvfs_model_path.exists():
            try:
                dvfs_model_info = joblib.load(dvfs_model_path)
                print(f"\nDVFS model loaded successfully")
                print(f"  DVFS model type: {dvfs_model_info['model_type']}")
                
                # Extract the actual model
                dvfs_model = dvfs_model_info['model']
                
                # Try to print coefficients based on model type
                if dvfs_model_info['model_type'] == 'basic_linear':
                    print(f"  Coefficients: {dvfs_model.coef_}")
                    print(f"  Intercept: {dvfs_model.intercept_}")
                elif dvfs_model_info['model_type'] == 'linear_with_clock':
                    print(f"  Coefficients: {dvfs_model.coef_}")
                    print(f"  Intercept: {dvfs_model.intercept_}")
                elif dvfs_model_info['model_type'] == 'cubic_dvfs':
                    print(f"  Coefficients: {dvfs_model.coef_}")
                    print(f"  Intercept: {dvfs_model.intercept_}")
                elif dvfs_model_info['model_type'] == 'interaction':
                    print(f"  Coefficients: {dvfs_model.coef_}")
                    print(f"  Intercept: {dvfs_model.intercept_}")
                elif dvfs_model_info['model_type'] == 'polynomial':
                    print(f"  Ridge coefficients: {dvfs_model.named_steps['ridge'].coef_}")
                    print(f"  Ridge intercept: {dvfs_model.named_steps['ridge'].intercept_}")
            except Exception as e:
                print(f"Error loading DVFS model: {e}")
    
    except ImportError:
        print("Could not import config module.")
    except Exception as e:
        print(f"Error checking model files: {e}")

def trace_dvfs_model_calculation(df):
    """Attempt to trace the DVFS model calculation to identify issues."""
    print("\n===== TRACING DVFS MODEL CALCULATION =====")
    
    # Grab an idle and active sample
    idle_sample = df[(df['sm_clock'] < 400) & (df['power_watts'] < 20)].iloc[0]
    active_sample = df[(df['sm_clock'] > 1000) & (df['power_watts'] > 50)].iloc[0]
    
    # Try different potential model formulations to see which might match
    
    # 1. Check if clock frequency might be misinterpreted as GHz instead of MHz
    print("\nTheory 1: Clock frequency unit mismatch (MHz vs GHz)")
    
    for sample_name, sample in [("Idle", idle_sample), ("Active", active_sample)]:
        print(f"\n{sample_name} Sample:")
        
        # Original values
        sm_clock_mhz = sample['sm_clock']
        actual_power = sample['power_watts']
        dvfs_prediction = sample['dvfs_prediction']
        
        # Convert MHz to GHz and recalculate features
        sm_clock_ghz = sm_clock_mhz / 1000
        max_clock_ghz = 1.95  # 1950 MHz = 1.95 GHz
        
        clock_ratio_ghz = sm_clock_ghz / max_clock_ghz
        f_cubed_normalized_ghz = (clock_ratio_ghz ** 3) * 100
        util_x_clock_ghz = sample['utilization'] * sm_clock_ghz
        
        print(f"  SM Clock: {sm_clock_mhz:.0f} MHz = {sm_clock_ghz:.3f} GHz")
        print(f"  Calculated Clock Ratio (GHz): {clock_ratio_ghz:.6f}")
        print(f"  Calculated f³ Normalized (GHz): {f_cubed_normalized_ghz:.6f}")
        print(f"  Calculated Util × Clock (GHz): {util_x_clock_ghz:.3f}")
    
    # 2. Check if there might be an issue with the normalization factor
    print("\nTheory 2: Incorrect normalization factor")
    
    for max_clock in [1950, 2000, 1500, 3000]:
        for sample_name, sample in [("Idle", idle_sample)]:
            sm_clock = sample['sm_clock']
            clock_ratio = sm_clock / max_clock
            f_cubed = (clock_ratio ** 3) * 100
            
            print(f"Using max_clock={max_clock} MHz:")
            print(f"  Clock Ratio: {clock_ratio:.6f}")
            print(f"  f³ Normalized: {f_cubed:.6f}")
    
    # 3. Check for calculation errors in the cubic term
    print("\nTheory 3: Cubic calculation issue")
    
    for sample_name, sample in [("Idle", idle_sample), ("Active", active_sample)]:
        sm_clock = sample['sm_clock']
        max_clock = 1950
        
        # Original calculation
        clock_ratio = sm_clock / max_clock
        f_cubed = (clock_ratio ** 3) * 100
        
        # Alternative calculations
        alt1 = (sm_clock ** 3) / (max_clock ** 3) * 100
        alt2 = (sm_clock / 1000) ** 3 * 100
        alt3 = sm_clock ** 3 / 1e9
        
        print(f"\n{sample_name} Sample (Clock = {sm_clock:.0f} MHz):")
        print(f"  Original f³: {f_cubed:.6f}")
        print(f"  Alt 1: (sm_clock³)/(max_clock³)×100 = {alt1:.6f}")
        print(f"  Alt 2: (sm_clock/1000)³×100 = {alt2:.6f}")
        print(f"  Alt 3: sm_clock³/1e9 = {alt3:.6f}")

def check_prediction_scaling(df):
    """Check if there's a simple scaling issue with the DVFS predictions."""
    print("\n===== CHECKING PREDICTION SCALING =====")
    
    # Calculate basic statistics on prediction ratios
    dvfs_to_actual_ratio = df['dvfs_prediction'] / df['power_watts']
    
    print(f"DVFS Prediction / Actual Power Ratio:")
    print(f"  Mean: {dvfs_to_actual_ratio.mean():.2f}x")
    print(f"  Median: {dvfs_to_actual_ratio.median():.2f}x")
    print(f"  Min: {dvfs_to_actual_ratio.min():.2f}x")
    print(f"  Max: {dvfs_to_actual_ratio.max():.2f}x")
    print(f"  Std Dev: {dvfs_to_actual_ratio.std():.2f}")
    
    # Check if scaling is consistent across power states
    print("\nRatio by clock frequency range:")
    
    # Group by clock frequency range
    df['clock_range'] = pd.cut(df['sm_clock'], 
                             bins=[0, 400, 1000, 2000],
                             labels=['Low', 'Medium', 'High'])
    
    for clock_range, group in df.groupby('clock_range'):
        ratio = group['dvfs_prediction'] / group['power_watts']
        print(f"  {clock_range} Clock ({len(group)} samples): {ratio.mean():.2f}x (±{ratio.std():.2f})")
    
    # Test if scaling back by a factor helps
    if dvfs_to_actual_ratio.mean() > 10:  # If predictions are way off
        test_factor = 1.0 / dvfs_to_actual_ratio.median()
        scaled_predictions = df['dvfs_prediction'] * test_factor
        scaled_error = df['power_watts'] - scaled_predictions
        scaled_abs_error = abs(scaled_error)
        
        print(f"\nTesting scaling correction factor: {test_factor:.6f}")
        print(f"  Original DVFS MAE: {df['dvfs_abs_error'].mean():.2f} W")
        print(f"  Scaled DVFS MAE: {scaled_abs_error.mean():.2f} W")
        print(f"  Improvement: {(1 - scaled_abs_error.mean() / df['dvfs_abs_error'].mean()) * 100:.2f}%")
        
        # Plot original vs scaled predictions
        plt.figure(figsize=(12, 6))
        
        plt.subplot(1, 2, 1)
        plt.scatter(df['power_watts'], df['dvfs_prediction'], alpha=0.5)
        plt.plot([0, df['power_watts'].max()], [0, df['power_watts'].max()], 'r--')
        plt.xlabel('Actual Power (W)')
        plt.ylabel('DVFS Prediction (W)')
        plt.title('Original DVFS Predictions')
        
        plt.subplot(1, 2, 2)
        plt.scatter(df['power_watts'], scaled_predictions, alpha=0.5)
        plt.plot([0, df['power_watts'].max()], [0, df['power_watts'].max()], 'r--')
        plt.xlabel('Actual Power (W)')
        plt.ylabel('Scaled DVFS Prediction (W)')
        plt.title(f'Scaled Predictions (×{test_factor:.6f})')
        
        plt.tight_layout()
        plt.savefig('power_plots/dvfs_scaling_test.png')
        print(f"Scaling test plot saved to power_plots/dvfs_scaling_test.png")

def analyze_feature_impacts(df):
    """Analyze how different features impact the DVFS model predictions."""
    print("\n===== ANALYZING FEATURE IMPACTS =====")
    
    # Calculate correlations
    correlations = df[['power_watts', 'utilization', 'sm_clock', 'dvfs_prediction']].corr()
    print("Correlation matrix:")
    print(correlations)
    
    # Try to visualize which feature has the strongest impact
    plt.figure(figsize=(14, 5))
    
    plt.subplot(1, 2, 1)
    plt.scatter(df['utilization'], df['dvfs_prediction'], alpha=0.5)
    plt.xlabel('Utilization (%)')
    plt.ylabel('DVFS Prediction (W)')
    plt.title('DVFS Prediction vs. Utilization')
    
    plt.subplot(1, 2, 2)
    plt.scatter(df['sm_clock'], df['dvfs_prediction'], alpha=0.5)
    plt.xlabel('SM Clock (MHz)')
    plt.ylabel('DVFS Prediction (W)')
    plt.title('DVFS Prediction vs. Clock Frequency')
    
    plt.tight_layout()
    plt.savefig('power_plots/dvfs_feature_impacts.png')
    print(f"Feature impact plot saved to power_plots/dvfs_feature_impacts.png")
    
    # Check if predictions are dominated by a single feature
    # Fit simple models to see if one feature explains most of the prediction
    from sklearn.linear_model import LinearRegression
    
    X_util = df[['utilization']].values
    X_clock = df[['sm_clock']].values
    y_pred = df['dvfs_prediction'].values
    
    util_model = LinearRegression().fit(X_util, y_pred)
    clock_model = LinearRegression().fit(X_clock, y_pred)
    
    util_r2 = util_model.score(X_util, y_pred)
    clock_r2 = clock_model.score(X_clock, y_pred)
    
    print(f"\nR² when explaining DVFS predictions:")
    print(f"  Utilization alone: {util_r2:.4f}")
    print(f"  Clock frequency alone: {clock_r2:.4f}")
    
    print(f"\nRegression coefficients:")
    print(f"  Utilization model: {util_model.coef_[0]:.4f} × Utilization + {util_model.intercept_:.4f}")
    print(f"  Clock model: {clock_model.coef_[0]:.4f} × Clock + {clock_model.intercept_:.4f}")

def main():
    """Main function to debug the DVFS model issues."""
    # Ensure the plots directory exists
    Path("power_plots").mkdir(exist_ok=True)
    
    # Load the data
    df = load_data()
    
    # Run the debugging analyses
    examine_sample_points(df)
    check_model_files()
    trace_dvfs_model_calculation(df)
    check_prediction_scaling(df)
    analyze_feature_impacts(df)
    
    print("\nDebugging complete. Check the output above for insights into the DVFS model issues.")

if __name__ == "__main__":
    main()
