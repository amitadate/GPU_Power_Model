#!/usr/bin/env python3


import os
import sys
import joblib
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime


sys.path.insert(0, str(Path(__file__).resolve().parent))


from file_utils import find_most_recent_power_data, create_timestamp

import config

def load_dvfs_model():
    """Load the best DVFS-aware power model."""
    try:
        # Try to load the simple linear model first if it exists
        simple_model_path = config.MODELS_DIR / "dvfs_simple_model.pkl"
        if simple_model_path.exists():
            model_info = joblib.load(simple_model_path)
            print(f"Loaded simple DVFS model ({model_info['model_type']}) from {simple_model_path}")
            return model_info
            
        # Otherwise load the best model
        best_model_path = config.MODELS_DIR / "dvfs_best_model.pkl"
        model_info = joblib.load(best_model_path)
        print(f"Loaded best DVFS model ({model_info['model_type']}) from {best_model_path}")
        return model_info
    except Exception as e:
        print(f"Error loading DVFS model: {e}")
        return None

def load_basic_model():
    """Load the basic linear power model."""
    try:
        basic_model_path = config.MODELS_DIR / config.MODEL_FILENAME
        model = joblib.load(basic_model_path)
        print(f"Loaded basic power model from {basic_model_path}")
        return model
    except Exception as e:
        print(f"Error loading basic model: {e}")
        return None

# def apply_dvfs_model_to_data(model_info, power_data):
    # """
    # Apply the DVFS-aware power model to the collected power data.
    # 
    # Args:
        # model_info: Dictionary containing model and model_type
        # power_data: DataFrame containing power monitoring data
        # 
    # Returns:
        # DataFrame with additional columns for model predictions
    # """
    #####Examine model coefficients
    # 
    # if model_info is None or power_data is None or power_data.empty:
        # print("No model or data available")
        # return power_data
    # 
    #####Make a copy of the DataFrame
    # df = power_data.copy()
    # 
    ####Get the model and model type
    # model = model_info['model']
    # model_type = model_info['model_type']
# 
# 
    # print(f"Model type: {model_type}")
    # if hasattr(model, 'coef_'):
        # print(f"Coefficients: {model.coef_}")
        # print(f"Intercept: {model.intercept_}")
    # elif hasattr(model, 'named_steps') and 'ridge' in model.named_steps:
        # print(f"Ridge coefficients: {model.named_steps['ridge'].coef_}")
        # print(f"Ridge intercept: {model.named_steps['ridge'].intercept_}")
    # 
    ######Maximum SM clock observed during training (for normalization)
    # max_sm_clock = 1950  # Typical max for RTX series
    # 
    ######Calculate derived features
    # df['clock_ratio'] = df['sm_clock'] / max_sm_clock
    # df['f_cubed_normalized'] = (df['clock_ratio'] ** 3) * 100
    # df['util_x_clock'] = df['utilization'] * df['sm_clock']
    # 
    #####Make predictions based on model type
    # try:
        # if model_type == 'basic_linear':
            # df['dvfs_prediction'] = model.predict(df[['utilization']].values)
            # df['dvfs_prediction'] = np.maximum(0, df['dvfs_prediction'])
            #######Cap predictions at a reasonable maximum (20% above TDP)
            # tdp = 250  # TDP for RTX 2080 Super
            # df['dvfs_prediction'] = np.minimum(df['dvfs_prediction'], tdp * 1.2)
        # elif model_type == 'linear_with_clock':
            # df['dvfs_prediction'] = model.predict(df[['utilization', 'sm_clock']].values)
            # df['dvfs_prediction'] = np.maximum(0, df['dvfs_prediction'])
            ######Cap predictions at a reasonable maximum (20% above TDP)
            # tdp = 250  # TDP for RTX 2080 Super
            # df['dvfs_prediction'] = np.minimum(df['dvfs_prediction'], tdp * 1.2)
        # elif model_type == 'cubic_dvfs':
            # df['dvfs_prediction'] = model.predict(df[['utilization', 'f_cubed_normalized']].values)
            # df['dvfs_prediction'] = np.maximum(0, df['dvfs_prediction'])
            ######Cap predictions at a reasonable maximum (20% above TDP)
            # tdp = 250  # TDP for RTX 2080 Super
            # df['dvfs_prediction'] = np.minimum(df['dvfs_prediction'], tdp * 1.2)
        # elif model_type == 'interaction':
            # df['dvfs_prediction'] = model.predict(df[['utilization', 'sm_clock', 'util_x_clock']].values)
            # df['dvfs_prediction'] = np.maximum(0, df['dvfs_prediction'])
            #####Cap predictions at a reasonable maximum (20% above TDP)
            # tdp = 250  # TDP for RTX 2080 Super
            # df['dvfs_prediction'] = np.minimum(df['dvfs_prediction'], tdp * 1.2)
        # elif model_type == 'polynomial':
            # df['dvfs_prediction'] = model.predict(df[['utilization', 'sm_clock']].values)
            # df['dvfs_prediction'] = np.maximum(0, df['dvfs_prediction'])
            #####Cap predictions at a reasonable maximum (20% above TDP)
            # tdp = 250  # TDP for RTX 2080 Super
            # df['dvfs_prediction'] = np.minimum(df['dvfs_prediction'], tdp * 1.2)
        # else:
            # print(f"Unknown model type: {model_type}")
            # return df
        # 
        #######Calculate error
        # df['dvfs_error'] = df['power_watts'] - df['dvfs_prediction']
        # df['dvfs_abs_error'] = df['dvfs_error'].abs()
        # 
        # print(f"Applied {model_type} model to data")
    # except Exception as e:
        # print(f"Error applying DVFS model: {e}")
    # 
    # return df

def apply_dvfs_model_to_data(model_info, power_data):

    if model_info is None or power_data is None or power_data.empty:
        print("No model or data available")
        return power_data
    
    # Make a copy of the DataFrame
    df = power_data.copy()
    
    # Get the model and model type
    model = model_info['model']
    model_type = model_info['model_type']
    
    print(f"Model type: {model_type}")
    if hasattr(model, 'coef_'):
        print(f"Coefficients: {model.coef_}")
        print(f"Intercept: {model.intercept_}")
    elif hasattr(model, 'named_steps') and 'ridge' in model.named_steps:
        print(f"Ridge coefficients: {model.named_steps['ridge'].coef_}")
        print(f"Ridge intercept: {model.named_steps['ridge'].intercept_}")
    
    # Maximum SM clock observed during training (for normalization)
    max_sm_clock = 2000  # Max for RTX 2080 Super
    
    # Calculate derived features
    df['clock_ratio'] = df['sm_clock'] / max_sm_clock
    # Ensure ratio is in valid range
    df['clock_ratio'] = df['clock_ratio'].clip(0, 1.0)
    
    df['f_cubed_normalized'] = (df['clock_ratio'] ** 3) * 10  # Reduced from 100 to 10
    df['util_x_clock'] = df['utilization'] * df['sm_clock']
    
    # Make predictions based on model type
    try:
        if model_type == 'basic_linear':
            df['dvfs_prediction'] = model.predict(df[['utilization']].values)
        elif model_type == 'linear_with_clock':
            df['dvfs_prediction'] = model.predict(df[['utilization', 'sm_clock']].values)
        elif model_type == 'cubic_dvfs':
            df['dvfs_prediction'] = model.predict(df[['utilization', 'f_cubed_normalized']].values)
        elif model_type == 'interaction':
            df['dvfs_prediction'] = model.predict(df[['utilization', 'sm_clock', 'util_x_clock']].values)
        elif model_type == 'polynomial':
            # For polynomial model, we need a fallback with safe coefficients, its still buggy
            try:
                # Use the standard prediction path first
                df['dvfs_prediction'] = model.predict(df[['utilization', 'sm_clock']].values)
            except Exception as e:
                print(f"Error with polynomial model prediction: {e}")
                # Fallback to manual prediction with safe coefficients
                print("Using manual prediction with safe coefficients")
                df['dvfs_prediction'] = 10 + 0.5 * df['utilization'] + 0.025 * df['sm_clock']
        else:
            print(f"Unknown model type: {model_type}")
            return df
        
        df['dvfs_prediction'] = np.maximum(0, df['dvfs_prediction'])
        # Cap predictions at a reasonable maximum (20% above TDP)
        tdp = 250  # TDP for RTX 2080 Super
        df['dvfs_prediction'] = np.minimum(df['dvfs_prediction'], tdp * 1.2)
        
        # Calculate error
        df['dvfs_error'] = df['power_watts'] - df['dvfs_prediction']
        df['dvfs_abs_error'] = df['dvfs_error'].abs()
        
        print(f"Applied {model_type} model to data")
    except Exception as e:
        print(f"Error applying DVFS model: {e}")
    
    return df


def apply_basic_model_to_data(model, power_data):
    if model is None or power_data is None or power_data.empty:
        print("No model or data available")
        return power_data
    
    # Make a copy of the DataFrame
    df = power_data.copy()
    
    # Make predictions
    try:
        df['basic_prediction'] = model.predict(df[['utilization']].values)
        
        # Calculate error
        df['basic_error'] = df['power_watts'] - df['basic_prediction']
        df['basic_abs_error'] = df['basic_error'].abs()
        
        print("Applied basic model to data")
    except Exception as e:
        print(f"Error applying basic model: {e}")
    
    return df

def analyze_model_predictions(df):

    if df is None or df.empty:
        print("No data to analyze")
        return None
    
    # Create a directory for the plots
    plots_dir = Path("power_plots")
    plots_dir.mkdir(exist_ok=True)
    
    # Create a timestamp for the filenames
    timestamp = create_timestamp()
    
    # Calculate model accuracy statistics for each kernel
    model_stats = []
    
    for kernel_name in df['kernel'].unique():
        if kernel_name == "None":
            continue
            
        # Filter data for this kernel
        kernel_df = df[df['kernel'] == kernel_name]
        
        # Check if prediction columns exist
        has_basic = 'basic_prediction' in kernel_df.columns
        has_dvfs = 'dvfs_prediction' in kernel_df.columns
        
        if not (has_basic or has_dvfs):
            continue
        
        # Calculate basic statistics
        stats = {
            'kernel': kernel_name,
            'samples': len(kernel_df),
            'avg_power': kernel_df['power_watts'].mean(),
            'avg_utilization': kernel_df['utilization'].mean(),
            'avg_clock': kernel_df['sm_clock'].mean()
        }
        
        # Add basic model statistics if available
        if has_basic:
            stats.update({
                'basic_prediction': kernel_df['basic_prediction'].mean(),
                'basic_mae': kernel_df['basic_abs_error'].mean(),
                'basic_rmse': np.sqrt((kernel_df['basic_error'] ** 2).mean()),
                'basic_error_pct': (kernel_df['basic_abs_error'] / kernel_df['power_watts']).mean() * 100
            })
        
        # Add DVFS model statistics if available
        if has_dvfs:
            stats.update({
                'dvfs_prediction': kernel_df['dvfs_prediction'].mean(),
                'dvfs_mae': kernel_df['dvfs_abs_error'].mean(),
                'dvfs_rmse': np.sqrt((kernel_df['dvfs_error'] ** 2).mean()),
                'dvfs_error_pct': (kernel_df['dvfs_abs_error'] / kernel_df['power_watts']).mean() * 100
            })
        
        model_stats.append(stats)
    
    # Create a DataFrame with the statistics
    stats_df = pd.DataFrame(model_stats) if model_stats else None
    
    # Save the statistics
    if stats_df is not None and not stats_df.empty:
        stats_file = f"power_results/model_prediction_stats_{timestamp}.csv"
        stats_df.to_csv(stats_file, index=False)
        print(f"Model prediction statistics saved to {stats_file}")
    
        # Create comparison plots for each kernel
        for kernel_name in df['kernel'].unique():
            if kernel_name == "None":
                continue
                
            kernel_df = df[df['kernel'] == kernel_name].copy()
            
            # Adjust timestamps to be relative to the start of the kernel
            min_time = kernel_df['timestamp'].min()
            kernel_df['relative_time'] = kernel_df['timestamp'] - min_time
            
            # Create power profile with predictions
            plt.figure(figsize=(12, 6))
            plt.plot(kernel_df['relative_time'], kernel_df['power_watts'], 'k-', label='Actual Power')
            
            if 'basic_prediction' in kernel_df:
                plt.plot(kernel_df['relative_time'], kernel_df['basic_prediction'], 'b--', label='Basic Model')
            
            if 'dvfs_prediction' in kernel_df:
                plt.plot(kernel_df['relative_time'], kernel_df['dvfs_prediction'], 'r--', label='DVFS Model')
            
            plt.xlabel('Time (seconds)')
            plt.ylabel('Power (Watts)')
            plt.title(f'Power Predictions for {kernel_name}')
            plt.legend()
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.tight_layout()
            
            prediction_plot = f"power_plots/{kernel_name}_predictions_{timestamp}.png"
            plt.savefig(prediction_plot)
            plt.close()
            
            # Create error histograms if both models are available
            if 'basic_abs_error' in kernel_df and 'dvfs_abs_error' in kernel_df:
                plt.figure(figsize=(10, 6))
                
                plt.hist(kernel_df['basic_error'], bins=20, alpha=0.5, label='Basic Model Error')
                plt.hist(kernel_df['dvfs_error'], bins=20, alpha=0.5, label='DVFS Model Error')
                
                plt.xlabel('Error (Watts)')
                plt.ylabel('Frequency')
                plt.title(f'Prediction Error Distribution for {kernel_name}')
                plt.legend()
                plt.grid(True, linestyle='--', alpha=0.7)
                plt.tight_layout()
                
                error_plot = f"power_plots/{kernel_name}_error_dist_{timestamp}.png"
                plt.savefig(error_plot)
                plt.close()
        
        # Create a summary comparison of model errors
        if stats_df is not None and 'basic_mae' in stats_df.columns and 'dvfs_mae' in stats_df.columns:
            plt.figure(figsize=(10, 6))
            
            width = 0.35
            x = np.arange(len(stats_df))
            
            plt.bar(x - width/2, stats_df['basic_mae'], width, label='Basic Model')
            plt.bar(x + width/2, stats_df['dvfs_mae'], width, label='DVFS Model')
            
            plt.xlabel('Kernel Implementation')
            plt.ylabel('Mean Absolute Error (Watts)')
            plt.title('Model Prediction Accuracy by Kernel')
            plt.xticks(x, stats_df['kernel'], rotation=45)
            plt.legend()
            plt.grid(True, axis='y', linestyle='--', alpha=0.7)
            plt.tight_layout()
            
            mae_plot = f"power_plots/model_mae_comparison_{timestamp}.png"
            plt.savefig(mae_plot)
            plt.close()
            
            # Create improvement percentage plot if we have both models
            if stats_df is not None and 'basic_mae' in stats_df.columns and 'dvfs_mae' in stats_df.columns:
                plt.figure(figsize=(10, 6))
                
                # Calculate improvement percentage
                stats_df['improvement_pct'] = (stats_df['basic_mae'] - stats_df['dvfs_mae']) / stats_df['basic_mae'] * 100
                
                plt.bar(stats_df['kernel'], stats_df['improvement_pct'])
                plt.xlabel('Kernel Implementation')
                plt.ylabel('Improvement (%)')
                plt.title('DVFS Model Improvement over Basic Model')
                plt.xticks(rotation=45)
                plt.grid(True, axis='y', linestyle='--', alpha=0.7)
                plt.tight_layout()
                
                improvement_plot = f"power_plots/model_improvement_{timestamp}.png"
                plt.savefig(improvement_plot)
                plt.close()



    # Add a hybrid model that combines basic and DVFS
    df['hybrid_prediction'] = df['basic_prediction'].copy()
    
    # Use DVFS prediction only for high clock frequencies
    high_clock_mask = df['sm_clock'] >= 1000
    df.loc[high_clock_mask, 'hybrid_prediction'] = df.loc[high_clock_mask, 'dvfs_prediction']
    
    # Calculate hybrid model errors
    df['hybrid_error'] = df['power_watts'] - df['hybrid_prediction']
    df['hybrid_abs_error'] = df['hybrid_error'].abs()
    
    # Add hybrid model to statistics for each kernel
    if stats_df is not None and not stats_df.empty:
        for kernel_name in df['kernel'].unique():
            if kernel_name == "None":
                continue
                
            kernel_df = df[df['kernel'] == kernel_name]
            
            # Skip if there's no data
            if kernel_df.empty:
                continue
                
            # Find corresponding row in stats_df
            row_idx = stats_df[stats_df['kernel'] == kernel_name].index
            if len(row_idx) == 0:
                continue
                
            # Add hybrid model statistics
            stats_df.loc[row_idx, 'hybrid_prediction'] = kernel_df['hybrid_prediction'].mean()
            stats_df.loc[row_idx, 'hybrid_mae'] = kernel_df['hybrid_abs_error'].mean()
            stats_df.loc[row_idx, 'hybrid_rmse'] = np.sqrt((kernel_df['hybrid_error'] ** 2).mean())
            stats_df.loc[row_idx, 'hybrid_error_pct'] = (kernel_df['hybrid_abs_error'] / kernel_df['power_watts']).mean() * 100
            
        # Also add hybrid improvement percentage
        if 'basic_mae' in stats_df.columns and 'hybrid_mae' in stats_df.columns:
            stats_df['hybrid_improvement_pct'] = (stats_df['basic_mae'] - stats_df['hybrid_mae']) / stats_df['basic_mae'] * 100
    
    print("\nModel prediction analysis completed.")
    print(f"Plots saved to the power_plots directory.")



    
    
    return stats_df

def main():
    data_path = find_most_recent_power_data()
    
    if data_path is None:
        print("Error: No power data CSV files found in power_results directory")
        print("Please run histogram_power_harness.py first")
        return
    
    print(f"Automatically selected most recent power data file: {data_path}")
    
    # Load the power data
    try:
        power_data = pd.read_csv(data_path)
        print(f"Loaded power data from {data_path}")
        print(f"Data contains {len(power_data)} samples across {power_data['kernel'].nunique()} kernels")
    except Exception as e:
        print(f"Error loading power data: {e}")
        return
    
    # Load the models
    dvfs_model_info = load_dvfs_model()
    basic_model = load_basic_model()
    
    # Apply the models to the data
    if basic_model:
        power_data = apply_basic_model_to_data(basic_model, power_data)
    
    if dvfs_model_info:
        power_data = apply_dvfs_model_to_data(dvfs_model_info, power_data)
    
    # Analyze the model predictions
    stats_df = analyze_model_predictions(power_data)
    
    # Save the enhanced data with predictions
    enhanced_data_path = data_path.parent / f"enhanced_{data_path.name}"
    power_data.to_csv(enhanced_data_path, index=False)
    print(f"Enhanced data with predictions saved to {enhanced_data_path}")
    
    # Print summary
    if stats_df is not None and not stats_df.empty:
        print("\nModel Prediction Summary:")
        print(stats_df.to_string(index=False))
        
        if 'improvement_pct' in stats_df.columns:
            avg_improvement = stats_df['improvement_pct'].mean()
            print(f"\nAverage DVFS model improvement: {avg_improvement:.2f}%")

if __name__ == "__main__":
    main()
