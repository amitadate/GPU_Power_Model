#!/usr/bin/env python3

import os
import sys
import time
import subprocess
import re
from pathlib import Path

def run_command_with_progress(command, description):
    print(f"\n{'=' * 80}")
    print(f"{description}")
    print(f"{'=' * 80}\n")
    

    return_code = subprocess.call(command, shell=True)
    
    if return_code != 0:
        print(f"Command failed with return code {return_code}")
    
    return return_code

def ensure_directories():
    directories = ["power_results", "power_plots", "energy_analysis", "models"]
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
    print("Created necessary directories")

def create_memory_model_trainer():
    script_path = Path("memory_model_trainer.py")
    
    with open(script_path, 'w') as f:
        f.write('''#!/usr/bin/env python3

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
    """
    Load the most recent enhanced power data file with memory metrics.
    """
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
        
        print("\\nModel comparison:")
        print("  Basic model and DVFS cubic model loaded for reference")
    except Exception as e:
        print(f"Could not load existing models for comparison: {e}")

if __name__ == "__main__":
    main()
''')
    
    # Make the script executable
    os.chmod(script_path, 0o755)
    return script_path

def create_memory_predictor_script():
    """Create a script to test the memory-aware power model."""
    script_path = Path("memory_aware_power_predictor.py")
    
    with open(script_path, 'w') as f:
        f.write('''#!/usr/bin/env python3


import os
import sys
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime

# Add the current directory to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent))

# Import from your existing GPU power modeling code
import config

def find_most_recent_file(directory, pattern):
    """Find the most recent file matching a pattern in a directory."""
    directory = Path(directory)
    matching_files = list(directory.glob(pattern))
    if not matching_files:
        return None
    return max(matching_files, key=lambda p: p.stat().st_mtime)

def load_enhanced_power_data():
    """Load the most recent enhanced power data with memory metrics."""
    # Find the most recent enhanced data file
    enhanced_data_file = find_most_recent_file("power_results", "enhanced_histogram_power_data_*.csv")
    if not enhanced_data_file:
        print("No enhanced power data file found.")
        sys.exit(1)
    
    # Load the data
    print(f"Loading data from {enhanced_data_file}")
    df = pd.read_csv(enhanced_data_file)
    
    return df

def load_memory_aware_model():
    """Load the memory-aware power model."""
    try:
        # Try the memory-aware cubic model first
        memory_model_path = config.MODELS_DIR / "memory_aware_cubic_model.pkl"
        if memory_model_path.exists():
            model_info = joblib.load(memory_model_path)
            print(f"Loaded memory-aware model ({model_info['model_type']}) from {memory_model_path}")
            return model_info
            
        # Try the memory-aware full feature model next
        memory_model_path = config.MODELS_DIR / "memory_aware_full_model.pkl"
        if memory_model_path.exists():
            model_info = joblib.load(memory_model_path)
            print(f"Loaded memory-aware model ({model_info['model_type']}) from {memory_model_path}")
            return model_info
            
        print("No memory-aware model found. Please run memory_model_trainer.py first.")
        return None
    except Exception as e:
        print(f"Error loading memory-aware model: {e}")
        return None

def load_basic_model():
    """Load the basic power model."""
    try:
        basic_model_path = config.MODELS_DIR / config.MODEL_FILENAME
        model = joblib.load(basic_model_path)
        print(f"Loaded basic power model from {basic_model_path}")
        return model
    except Exception as e:
        print(f"Error loading basic model: {e}")
        return None

def load_dvfs_model():
    """Load the DVFS power model."""
    try:
        # Try different DVFS model types
        model_paths = [
            "dvfs_cubic_dvfs_model.pkl",
            "dvfs_linear_with_clock_model.pkl",
            "dvfs_simple_model.pkl"
        ]
        
        for model_path in model_paths:
            full_path = config.MODELS_DIR / model_path
            if full_path.exists():
                model_info = joblib.load(full_path)
                # For models with 'model_type'
                if isinstance(model_info, dict) and 'model_type' in model_info:
                    print(f"Loaded DVFS model ({model_info['model_type']}) from {full_path}")
                    return model_info
                else:
                    # For simpler models (like LinearRegression)
                    print(f"Loaded simple DVFS model from {full_path}")
                    return {'model': model_info, 'model_type': 'simple_linear'}
        
        print("No DVFS model found.")
        return None
    except Exception as e:
        print(f"Error loading DVFS model: {e}")
        return None

def apply_memory_aware_model(model_info, power_data):
    if model_info is None or power_data is None or power_data.empty:
        print("No model or data available")
        return power_data
    
    # Make a copy of the DataFrame
    df = power_data.copy()
    
    # Get the model and model type
    model = model_info['model']
    model_type = model_info['model_type']
    
    print(f"Model type: {model_type}")
    
    # Apply predictions based on model type
    try:
        if model_type == 'memory_aware_cubic_dvfs':
            # Create derived features
            df['clock_ghz'] = df['sm_clock'] / 1000.0  # Convert to GHz
            df['f_cubed'] = df['clock_ghz'] ** 3
            df['f_linear'] = df['clock_ghz']
            
            # Add memory factor if available
            if 'memory_bandwidth_pct' in df.columns:
                df['mem_factor'] = 1.0 + (df['memory_bandwidth_pct'] / 100.0) * 0.5
            else:
                df['mem_factor'] = 1.0  # Default if no memory data
                
            df['f_cubed_mem'] = df['f_cubed'] * df['mem_factor']
            
            # Predict power
            df['memory_aware_prediction'] = model.predict(df[['utilization', 'f_cubed_mem', 'f_linear']].values)
            
        elif model_type == 'memory_aware_polynomial':
            # Create derived features
            df['clock_ghz'] = df['sm_clock'] / 1000.0  # Convert to GHz
            
            # Ensure memory_bandwidth_pct exists
            if 'memory_bandwidth_pct' not in df.columns:
                df['memory_bandwidth_pct'] = 0.0  # Default if no memory data
            
            # Predict power
            df['memory_aware_prediction'] = model.predict(df[['utilization', 'clock_ghz', 'memory_bandwidth_pct']].values)
            
        else:
            print(f"Unknown memory model type: {model_type}")
            return df
        
        # Ensure predictions are non-negative
        df['memory_aware_prediction'] = np.maximum(0, df['memory_aware_prediction'])
        
        # Cap predictions at a reasonable maximum (20% above TDP)
        tdp = 250  # TDP for RTX 2080 Super
        df['memory_aware_prediction'] = np.minimum(df['memory_aware_prediction'], tdp * 1.2)
        
        # Calculate error metrics
        df['memory_aware_error'] = df['power_watts'] - df['memory_aware_prediction']
        df['memory_aware_abs_error'] = df['memory_aware_error'].abs()
        
        print(f"Applied {model_type} model to data")
    except Exception as e:
        print(f"Error applying memory-aware model: {e}")
        print(f"Exception details: {str(e)}")
        import traceback
        traceback.print_exc()
    
    return df

def apply_basic_model(model, power_data):
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

def apply_dvfs_model(model_info, power_data):
    if model_info is None or power_data is None or power_data.empty:
        print("No model or data available")
        return power_data
    
    # Make a copy of the DataFrame
    df = power_data.copy()
    
    # Get the model and model type
    model = model_info['model']
    model_type = model_info['model_type']
    
    print(f"Model type: {model_type}")
    
    # Apply predictions based on model type
    try:
        if model_type == 'cubic_dvfs':
            # Create derived features
            df['clock_ratio'] = df['sm_clock'] / 2000.0  # Normalize to typical max
            df['clock_ratio'] = df['clock_ratio'].clip(0, 1.0)  # Ensure in valid range
            df['f_cubed_normalized'] = (df['clock_ratio'] ** 3) * 10
            
            # Predict power
            df['dvfs_prediction'] = model.predict(df[['utilization', 'f_cubed_normalized']].values)
        
        elif model_type == 'linear_with_clock':
            df['dvfs_prediction'] = model.predict(df[['utilization', 'sm_clock']].values)
            
        elif model_type == 'simple_linear':
            # For simple LinearRegression model
            try:
                # Try with both utilization and sm_clock
                df['dvfs_prediction'] = model.predict(df[['utilization', 'sm_clock']].values)
            except:
                # Fall back to just utilization if the above fails
                df['dvfs_prediction'] = model.predict(df[['utilization']].values)
        
        else:
            print(f"Unknown DVFS model type: {model_type}")
            return df
        
        # Ensure predictions are non-negative
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
        print(f"Exception details: {str(e)}")
        import traceback
        traceback.print_exc()
    
    return df

def compare_models(df):
    """Compare the different power models."""
    models_to_compare = []
    
    if 'basic_abs_error' in df.columns:
        models_to_compare.append('basic')
    
    if 'dvfs_abs_error' in df.columns:
        models_to_compare.append('dvfs')
    
    if 'memory_aware_abs_error' in df.columns:
        models_to_compare.append('memory_aware')
    
    if not models_to_compare:
        print("No models to compare")
        return
    
    # Calculate overall error metrics
    print("\\n===== OVERALL MODEL ERROR METRICS =====")
    overall_stats = pd.DataFrame(columns=['Model', 'MAE (W)', 'RMSE (W)', 'Error (%)'])
    
    for model_name in models_to_compare:
        mae = df[f'{model_name}_abs_error'].mean()
        rmse = np.sqrt((df[f'{model_name}_error'] ** 2).mean())
        error_pct = (df[f'{model_name}_abs_error'] / df['power_watts']).mean() * 100
        
        overall_stats = overall_stats.append({
            'Model': model_name.replace('_', ' ').title(),
            'MAE (W)': mae,
            'RMSE (W)': rmse,
            'Error (%)': error_pct
        }, ignore_index=True)
    
    print(overall_stats.to_string(index=False))
    
    # Calculate per-kernel metrics
    print("\\n===== MODEL ERROR BY KERNEL =====")
    
    kernel_stats = []
    for kernel_name in df['kernel'].unique():
        if kernel_name == 'None':
            continue
            
        kernel_df = df[df['kernel'] == kernel_name]
        
        for model_name in models_to_compare:
            mae = kernel_df[f'{model_name}_abs_error'].mean()
            rmse = np.sqrt((kernel_df[f'{model_name}_error'] ** 2).mean())
            error_pct = (kernel_df[f'{model_name}_abs_error'] / kernel_df['power_watts']).mean() * 100
            
            kernel_stats.append({
                'Kernel': kernel_name,
                'Model': model_name.replace('_', ' ').title(),
                'Power (W)': kernel_df['power_watts'].mean(),
                'MAE (W)': mae,
                'Error (%)': error_pct
            })
    
    kernel_stats_df = pd.DataFrame(kernel_stats)
    print(kernel_stats_df.to_string(index=False))
    
    # Save the model comparison results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = f"power_results/model_comparison_{timestamp}.csv"
    kernel_stats_df.to_csv(results_file, index=False)
    
    # Create visualizations
    plots_dir = Path("power_plots")
    plots_dir.mkdir(exist_ok=True)
    
    # 1. Overall error comparison
    plt.figure(figsize=(10, 6))
    bars = plt.bar(overall_stats['Model'], overall_stats['MAE (W)'], color=['blue', 'green', 'red'][:len(models_to_compare)])
    
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                f'{height:.2f}', ha='center', va='bottom')
    
    plt.ylabel('Mean Absolute Error (W)')
    plt.title('Power Model Comparison')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    comparison_plot = plots_dir / f"model_comparison_{timestamp}.png"
    plt.savefig(comparison_plot)
    plt.close()
    
    # 2. Error by kernel for each model
    plt.figure(figsize=(14, 8))
    
    # Prepare data for grouped bar chart
    kernels = sorted(kernel_stats_df['Kernel'].unique())
    models = sorted(kernel_stats_df['Model'].unique())
    
    x = np.arange(len(kernels))
    width = 0.8 / len(models)
    
    for i, model in enumerate(models):
        model_data = kernel_stats_df[kernel_stats_df['Model'] == model]
        
        # Sort data to match kernels order
        sorted_data = pd.DataFrame(index=kernels, columns=['Error (%)'])
        for _, row in model_data.iterrows():
            sorted_data.loc[row['Kernel']] = row['Error (%)']
        
        plt.bar(x + (i - len(models)/2 + 0.5) * width, sorted_data['Error (%)'], 
                width, label=model)
    
    plt.xlabel('Kernel')
    plt.ylabel('Error (%)')
    plt.title('Model Error by Kernel')
    plt.xticks(x, kernels, rotation=45)
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    kernel_error_plot = plots_dir / f"kernel_model_error_{timestamp}.png"
    plt.savefig(kernel_error_plot)
    plt.close()
    
    # 3. Power trace with predictions for a sample kernel
    strategy1_df = df[df['kernel'] == 'Strategy1'].copy()
    if not strategy1_df.empty:
        plt.figure(figsize=(12, 6))
        
        plt.plot(strategy1_df['timestamp'], strategy1_df['power_watts'], 'k-', label='Actual Power')
        
        for model_name in models_to_compare:
            prediction_col = f'{model_name}_prediction'
            if prediction_col in strategy1_df.columns:
                plt.plot(strategy1_df['timestamp'], strategy1_df[prediction_col], '--', 
                        label=f'{model_name.replace("_", " ").title()} Model')
        
        if 'memory_bandwidth_pct' in strategy1_df.columns:
            # Add memory bandwidth on secondary axis
            ax2 = plt.twinx()
            ax2.plot(strategy1_df['timestamp'], strategy1_df['memory_bandwidth_pct'], 
                    'g:', alpha=0.6, label='Memory Bandwidth')
            ax2.set_ylabel('Memory Bandwidth (%)', color='g')
            ax2.tick_params(axis='y', labelcolor='g')
        
        plt.xlabel('Time (s)')
        plt.ylabel('Power (W)')
        plt.title('Power Predictions for Strategy1')
        plt.legend(loc='upper left')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        
        trace_plot = plots_dir / f"strategy1_predictions_{timestamp}.png"
        plt.savefig(trace_plot)
        plt.close()
    
    print(f"\\nModel comparison visualizations saved to:")
    print(f"  - {comparison_plot}")
    print(f"  - {kernel_error_plot}")
    if not strategy1_df.empty:
        print(f"  - {trace_plot}")
    
    return results_file

def main():
    # Load power data
    power_data = load_enhanced_power_data()
    
    # Load models
    memory_model = load_memory_aware_model()
    basic_model = load_basic_model()
    dvfs_model = load_dvfs_model()
    
    # Apply models to the data
    df = power_data.copy()
    
    if basic_model:
        df = apply_basic_model(basic_model, df)
    
    if dvfs_model:
        df = apply_dvfs_model(dvfs_model, df)
    
    if memory_model:
        df = apply_memory_aware_model(memory_model, df)
    
    # Compare model results
    results_file = compare_models(df)
    
    # Save the enhanced data with all predictions
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"power_results/model_predictions_{timestamp}.csv"
    df.to_csv(output_file, index=False)
    print(f"\\nPrediction results saved to {output_file}")

if __name__ == "__main__":
    main()
''')
    
    # Make the script executable
    os.chmod(script_path, 0o755)
    return script_path

def main():
    """Run the complete pipeline using the latest optimized code."""
    # Create required directories
    ensure_directories()
    
    # Create the memory model trainer script
    memory_model_trainer = create_memory_model_trainer()
    print(f"Created memory model trainer script: {memory_model_trainer}")
    
    # Create the memory predictor script
    memory_predictor = create_memory_predictor_script()
    print(f"Created memory-aware predictor script: {memory_predictor}")
    
    # Step 1: Run the enhanced power measurement harness
    run_command_with_progress("python enhanced_histogram_power_harness.py", 
                "Step 1: Running histogram kernel with enhanced power and memory monitoring")
    
    # Let the system stabilize after the first run
    print("\nWaiting for system to stabilize...")
    time.sleep(5)
    
    # Step 2: Train the memory-aware power model
    run_command_with_progress(f"python {memory_model_trainer}", 
                "Step 2: Training memory-aware power model")
    
    # Step 3: Apply all power models (including new memory-aware model) to the data
    run_command_with_progress(f"python {memory_predictor}", 
                "Step 3: Applying memory-aware power model and comparing with other models")
    
    # Step 4: Apply the original power predictor for comparison
    run_command_with_progress("python histogram_power_predictor_auto.py", 
                "Step 4: Applying original power models for comparison")
    
    # Step 5: Analyze energy efficiency
    run_command_with_progress("python histogram_energy_analyzer_auto.py", 
                "Step 5: Analyzing energy efficiency metrics")
    
    # Step 6: Analyze model prediction errors
    run_command_with_progress("python analyze_model_errors.py",
                "Step 6: Analyzing power model prediction errors")
    
    # Step 7: Run the optimization technique analysis
    run_command_with_progress("python kernel_energy_efficiency_analysis.py",
                "Step 7: Analyzing optimization techniques for energy efficiency")
    
    print(f"\n{'=' * 80}")
    print("Enhanced GPU Power Analysis Pipeline Complete!")
    print(f"{'=' * 80}\n")
    
    print("Results are available in:")
    print("  - power_results/ (Raw data, statistics, and model comparisons)")
    print("  - power_plots/ (Power, memory, and model prediction visualizations)")
    print("  - energy_analysis/ (Energy efficiency analysis and visualizations)")
    print("  - models/ (New memory-aware power model)")

if __name__ == "__main__":
    main()
