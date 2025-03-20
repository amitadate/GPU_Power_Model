#!/bin/bash

# Complete Histogram Power Model Script
# This script collects data, builds power models, and generates visualizations for the histogram workload

# Set paths - update these to match your environment
LAB3_DIR="/home/asa5078/368/experiment_lab3/pushed_lab3/labs"
cd $LAB3_DIR

# Create output directory with timestamp
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="histogram_power_model_${TIMESTAMP}"
mkdir -p $OUTPUT_DIR

echo "===== Histogram Power Model Analysis ====="
echo "Output directory: $OUTPUT_DIR"

# ----- DATA COLLECTION PHASE -----
echo ""
echo "PHASE 1: DATA COLLECTION"
echo "========================="

# Log system information
echo "Collecting system information..."
{
    echo "System Information:"
    nvidia-smi --query-gpu=gpu_name,driver_version,vbios_version,memory.total,power.default_limit,power.min_limit,power.max_limit --format=csv,noheader
    echo ""
    lscpu | grep "Model name\|CPU MHz\|CPU(s)\|Thread(s)"
} > "${OUTPUT_DIR}/system_info.txt"

# Function to run a single histogram strategy
run_strategy() {
    local strategy=$1
    local strategy_name
    
    case $strategy in
        0) strategy_name="Baseline" ;;
        1) strategy_name="PrivateHist" ;;
        2) strategy_name="SharedMem" ;;
        3) strategy_name="InputTile" ;;
        4) strategy_name="BankConflict" ;;
        5) strategy_name="Coalesced" ;;
        6) strategy_name="LargeTile" ;;
        7) strategy_name="LinearIndexing" ;;
        8) strategy_name="LocalHisto" ;;
        9) strategy_name="LocalBins" ;;
        *) strategy_name="Unknown" ;;
    esac
    
    echo ""
    echo "Running Strategy $strategy ($strategy_name)"
    
    # Prepare output files
    METRICS_FILE="${OUTPUT_DIR}/strategy_${strategy}_metrics.csv"
    OUTPUT_FILE="${OUTPUT_DIR}/strategy_${strategy}_output.txt"
    
    # Start power monitoring
    echo "Starting power monitoring..."
    nvidia-smi --query-gpu=timestamp,utilization.gpu,utilization.memory,power.draw,temperature.gpu,clocks.current.sm,clocks.current.memory,memory.used,memory.total,pstate --format=csv -l 0.05 > "$METRICS_FILE" &
    NVIDIA_SMI_PID=$!
    
    # Let monitoring start
    sleep 1
    
    # Record start time
    START_TIME=$(date +%s.%N)
    echo "Start time: $START_TIME" > "$OUTPUT_FILE"
    
    # Run the histogram program with the specific strategy
    # This assumes the opt_2dhisto_strategy function in opt_2dhisto.cu is properly implemented
    STRATEGY=$strategy ./bin/linux/release/lab3 >> "$OUTPUT_FILE" 2>&1
    
    # Record end time
    END_TIME=$(date +%s.%N)
    echo "End time: $END_TIME" >> "$OUTPUT_FILE"
    DURATION=$(echo "$END_TIME - $START_TIME" | bc)
    echo "Duration: $DURATION seconds" >> "$OUTPUT_FILE"
    
    # Stop power monitoring
    kill $NVIDIA_SMI_PID
    
    # Extract timing information
    grep "Timing" "$OUTPUT_FILE" > "${OUTPUT_DIR}/strategy_${strategy}_timing.txt"
    
    echo "Strategy $strategy complete"
}

# Run all strategies (skip strategy 1 as it's too slow)
echo "Running histogram strategies..."
for strategy in 0 2 3 4 5 6 7 8 9; do
    run_strategy $strategy
    # Cool down period
    echo "Cooling down for 5 seconds..."
    sleep 5
done

# Create CSV summary file for the collected data
echo "Creating data summary file..."
{
    echo "Strategy,ExecutionTime,AvgGPUUtil,AvgMemUtil,AvgPower,MaxPower,AvgTemp,AvgSMClock,AvgMemClock,MemoryUsedMB"
    
    # Process each strategy's data
    for strategy in 0 2 3 4 5 6 7 8 9; do
        METRICS_FILE="${OUTPUT_DIR}/strategy_${strategy}_metrics.csv"
        OUTPUT_FILE="${OUTPUT_DIR}/strategy_${strategy}_output.txt"
        
        # Extract timing info
        EXEC_TIME=$(grep "Clock Time" "$OUTPUT_FILE" | tail -n 1 | awk '{print $5}')
        
        # Calculate metrics
        AVG_GPU_UTIL=$(tail -n +2 "$METRICS_FILE" | awk -F, '{sum+=$2} END {print sum/NR}')
        AVG_MEM_UTIL=$(tail -n +2 "$METRICS_FILE" | awk -F, '{sum+=$3} END {print sum/NR}')
        AVG_POWER=$(tail -n +2 "$METRICS_FILE" | awk -F, '{sum+=$4} END {print sum/NR}')
        MAX_POWER=$(tail -n +2 "$METRICS_FILE" | awk -F, '{if($4>max) max=$4} END {print max}')
        AVG_TEMP=$(tail -n +2 "$METRICS_FILE" | awk -F, '{sum+=$5} END {print sum/NR}')
        AVG_SM_CLOCK=$(tail -n +2 "$METRICS_FILE" | awk -F, '{sum+=$6} END {print sum/NR}')
        AVG_MEM_CLOCK=$(tail -n +2 "$METRICS_FILE" | awk -F, '{sum+=$7} END {print sum/NR}')
        MEM_USED=$(tail -n +2 "$METRICS_FILE" | awk -F, '{sum+=$8} END {print sum/NR/1024}')
        
        echo "$strategy,$EXEC_TIME,$AVG_GPU_UTIL,$AVG_MEM_UTIL,$AVG_POWER,$MAX_POWER,$AVG_TEMP,$AVG_SM_CLOCK,$AVG_MEM_CLOCK,$MEM_USED"
    done
} > "${OUTPUT_DIR}/histogram_data_summary.csv"

# ----- MODEL BUILDING PHASE -----
echo ""
echo "PHASE 2: POWER MODEL BUILDING"
echo "============================"

# Create a Python script to build the power models
cat > "${OUTPUT_DIR}/build_power_models.py" << 'PYTHON_SCRIPT'
#!/usr/bin/env python3

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score
import sys
import os

# Set plotting style
plt.style.use('ggplot')

def load_data(csv_file):
    """Load and prepare data from the CSV summary file"""
    print(f"Loading data from {csv_file}")
    data = pd.read_csv(csv_file)
    
    # Basic data cleaning and preprocessing
    for col in data.columns:
        if data[col].dtype == object:
            try:
                data[col] = pd.to_numeric(data[col])
            except:
                pass
    
    print(f"Data loaded: {len(data)} rows with {len(data.columns)} columns")
    print(data.head())
    return data

def build_basic_power_model(data):
    """
    Build the basic power model: P = α × utilization + β
    """
    print("\n3.1 Building Basic Power Model")
    
    X = data[['AvgGPUUtil']].values
    y = data['AvgPower'].values
    
    # Fit linear model
    model = LinearRegression()
    model.fit(X, y)
    
    # Get coefficients
    alpha = model.coef_[0]
    beta = model.intercept_
    
    # Calculate predictions and metrics
    y_pred = model.predict(X)
    rmse = np.sqrt(mean_squared_error(y, y_pred))
    r2 = r2_score(y, y_pred)
    
    print(f"Basic Power Model: P = {alpha:.4f} × utilization + {beta:.4f}")
    print(f"RMSE: {rmse:.4f} W, R²: {r2:.4f}")
    
    # Visualize the model
    plt.figure(figsize=(10, 6))
    plt.scatter(X, y, color='blue', alpha=0.6, label='Actual')
    
    # Plot the regression line
    x_line = np.linspace(X.min(), X.max(), 100)
    y_line = alpha * x_line + beta
    plt.plot(x_line, y_line, color='red', label=f'P = {alpha:.4f} × util + {beta:.4f}')
    
    plt.xlabel('GPU Utilization (%)')
    plt.ylabel('Power (W)')
    plt.title('Basic Power Model: Power vs. GPU Utilization')
    plt.legend()
    plt.grid(True)
    plt.savefig('basic_power_model.png', dpi=300, bbox_inches='tight')
    
    return {'alpha': alpha, 'beta': beta, 'rmse': rmse, 'r2': r2, 'y_pred': y_pred}

def build_dvfs_linear_model(data):
    """
    Build the DVFS-aware linear model: P = α × utilization + β × f + γ
    """
    print("\n3.2.1 Building DVFS-Aware Linear Power Model")
    
    X = data[['AvgGPUUtil', 'AvgSMClock']].values
    y = data['AvgPower'].values
    
    # Fit linear model
    model = LinearRegression()
    model.fit(X, y)
    
    # Get coefficients
    alpha = model.coef_[0]
    beta = model.coef_[1]
    gamma = model.intercept_
    
    # Calculate predictions and metrics
    y_pred = model.predict(X)
    rmse = np.sqrt(mean_squared_error(y, y_pred))
    r2 = r2_score(y, y_pred)
    
    print(f"DVFS Linear Model: P = {alpha:.4f} × utilization + {beta:.6f} × f + {gamma:.4f}")
    print(f"RMSE: {rmse:.4f} W, R²: {r2:.4f}")
    
    # Visualize the model - 3D plot
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    ax.scatter(data['AvgGPUUtil'], data['AvgSMClock'], data['AvgPower'], 
               c='blue', marker='o', alpha=0.6, label='Actual')
    
    # Create a mesh grid for the prediction surface
    x_surf = np.linspace(data['AvgGPUUtil'].min(), data['AvgGPUUtil'].max(), 20)
    y_surf = np.linspace(data['AvgSMClock'].min(), data['AvgSMClock'].max(), 20)
    x_surf, y_surf = np.meshgrid(x_surf, y_surf)
    
    # Calculate the predicted power values
    z_surf = gamma + alpha * x_surf + beta * y_surf
    
    # Plot the prediction surface
    ax.plot_surface(x_surf, y_surf, z_surf, alpha=0.3, color='red')
    
    ax.set_xlabel('GPU Utilization (%)')
    ax.set_ylabel('SM Clock (MHz)')
    ax.set_zlabel('Power (W)')
    ax.set_title('DVFS-Aware Linear Power Model')
    
    plt.savefig('dvfs_linear_model.png', dpi=300, bbox_inches='tight')
    
    return {'alpha': alpha, 'beta': beta, 'gamma': gamma, 'rmse': rmse, 'r2': r2, 'y_pred': y_pred}

def build_dvfs_cubic_model(data):
    """
    Build the DVFS-aware cubic model: P = α × utilization + β × (f/fmax)³ + γ
    """
    print("\n3.2.2 Building DVFS-Aware Cubic Power Model")
    
    # Find maximum frequency
    f_max = data['AvgSMClock'].max()
    
    # Create the cubic feature for clock frequency
    data['f_cubic'] = (data['AvgSMClock'] / f_max) ** 3
    
    X = data[['AvgGPUUtil', 'f_cubic']].values
    y = data['AvgPower'].values
    
    # Fit linear model
    model = LinearRegression()
    model.fit(X, y)
    
    # Get coefficients
    alpha = model.coef_[0]
    beta = model.coef_[1]
    gamma = model.intercept_
    
    # Calculate predictions and metrics
    y_pred = model.predict(X)
    rmse = np.sqrt(mean_squared_error(y, y_pred))
    r2 = r2_score(y, y_pred)
    
    print(f"DVFS Cubic Model: P = {alpha:.4f} × utilization + {beta:.4f} × (f/fmax)³ + {gamma:.4f}")
    print(f"RMSE: {rmse:.4f} W, R²: {r2:.4f}")
    print(f"Maximum Clock: {f_max:.0f} MHz")
    
    # Visualize the model - 3D plot
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    ax.scatter(data['AvgGPUUtil'], data['f_cubic'], data['AvgPower'], 
               c='blue', marker='o', alpha=0.6, label='Actual')
    
    # Create a mesh grid for the prediction surface
    x_surf = np.linspace(data['AvgGPUUtil'].min(), data['AvgGPUUtil'].max(), 20)
    y_surf = np.linspace(data['f_cubic'].min(), data['f_cubic'].max(), 20)
    x_surf, y_surf = np.meshgrid(x_surf, y_surf)
    
    # Calculate the predicted power values
    z_surf = gamma + alpha * x_surf + beta * y_surf
    
    # Plot the prediction surface
    ax.plot_surface(x_surf, y_surf, z_surf, alpha=0.3, color='red')
    
    ax.set_xlabel('GPU Utilization (%)')
    ax.set_ylabel('Normalized Cubic Frequency (f/fmax)³')
    ax.set_zlabel('Power (W)')
    ax.set_title('DVFS-Aware Cubic Power Model')
    
    plt.savefig('dvfs_cubic_model.png', dpi=300, bbox_inches='tight')
    
    return {'alpha': alpha, 'beta': beta, 'gamma': gamma, 'f_max': f_max, 'rmse': rmse, 'r2': r2, 'y_pred': y_pred}

def build_dvfs_interaction_model(data):
    """
    Build the DVFS-aware interaction model: P = α × utilization + β × f + δ × (utilization × f) + γ
    """
    print("\n3.2.3 Building DVFS-Aware Interaction Power Model")
    
    # Create interaction term
    data['util_f_interaction'] = data['AvgGPUUtil'] * data['AvgSMClock']
    
    X = data[['AvgGPUUtil', 'AvgSMClock', 'util_f_interaction']].values
    y = data['AvgPower'].values
    
    # Fit linear model
    model = LinearRegression()
    model.fit(X, y)
    
    # Get coefficients
    alpha = model.coef_[0]
    beta = model.coef_[1]
    delta = model.coef_[2]
    gamma = model.intercept_
    
    # Calculate predictions and metrics
    y_pred = model.predict(X)
    rmse = np.sqrt(mean_squared_error(y, y_pred))
    r2 = r2_score(y, y_pred)
    
    print(f"DVFS Interaction Model: P = {alpha:.4f} × util + {beta:.6f} × f + {delta:.8f} × (util × f) + {gamma:.4f}")
    print(f"RMSE: {rmse:.4f} W, R²: {r2:.4f}")
    
    # Visualize actual vs predicted values
    plt.figure(figsize=(10, 6))
    plt.scatter(y, y_pred, alpha=0.6)
    
    # Plot the ideal line (y=x)
    min_val = min(y.min(), y_pred.min())
    max_val = max(y.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='Ideal')
    
    plt.xlabel('Actual Power (W)')
    plt.ylabel('Predicted Power (W)')
    plt.title('DVFS Interaction Model: Actual vs Predicted Power')
    plt.legend()
    plt.grid(True)
    plt.savefig('dvfs_interaction_model.png', dpi=300, bbox_inches='tight')
    
    return {'alpha': alpha, 'beta': beta, 'delta': delta, 'gamma': gamma, 'rmse': rmse, 'r2': r2, 'y_pred': y_pred}

def build_memory_cubic_model(data):
    """
    Build the memory-aware cubic model: P = α × utilization + β × f³ × M + τ × f + γ
    """
    print("\n3.3.1 Building Memory-Aware Cubic Power Model")
    
    # Create memory factor
    memory_util = data['AvgMemUtil'] / 100  # Convert to 0-1 scale
    data['M'] = 1 + 0.5 * memory_util
    
    # Create cubic term with memory factor
    data['f_cubic_mem'] = (data['AvgSMClock'] ** 3) * data['M'] / 1e9  # Scale down for numerical stability
    
    X = data[['AvgGPUUtil', 'f_cubic_mem', 'AvgSMClock']].values
    y = data['AvgPower'].values
    
    # Fit linear model
    model = LinearRegression()
    model.fit(X, y)
    
    # Get coefficients
    alpha = model.coef_[0]
    beta = model.coef_[1]
    tau = model.coef_[2]
    gamma = model.intercept_
    
    # Calculate predictions and metrics
    y_pred = model.predict(X)
    rmse = np.sqrt(mean_squared_error(y, y_pred))
    r2 = r2_score(y, y_pred)
    
    print(f"Memory-Aware Cubic Model: P = {alpha:.4f} × util + {beta:.8f} × f³ × M + {tau:.6f} × f + {gamma:.4f}")
    print(f"RMSE: {rmse:.4f} W, R²: {r2:.4f}")
    
    # Visualize the model with a multi-variable scatter plot
    plt.figure(figsize=(15, 10))
    
    # 1. Utilization vs Power
    plt.subplot(2, 2, 1)
    plt.scatter(data['AvgGPUUtil'], data['AvgPower'], alpha=0.6)
    plt.xlabel('GPU Utilization (%)')
    plt.ylabel('Power (W)')
    plt.title('Utilization vs Power')
    
    # 2. Frequency vs Power
    plt.subplot(2, 2, 2)
    plt.scatter(data['AvgSMClock'], data['AvgPower'], alpha=0.6)
    plt.xlabel('SM Clock (MHz)')
    plt.ylabel('Power (W)')
    plt.title('Frequency vs Power')
    
    # 3. Memory Utilization vs Power
    plt.subplot(2, 2, 3)
    plt.scatter(data['AvgMemUtil'], data['AvgPower'], alpha=0.6)
    plt.xlabel('Memory Utilization (%)')
    plt.ylabel('Power (W)')
    plt.title('Memory Utilization vs Power')
    
    # 4. Actual vs Predicted
    plt.subplot(2, 2, 4)
    plt.scatter(y, y_pred, alpha=0.6)
    min_val = min(y.min(), y_pred.min())
    max_val = max(y.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='Ideal')
    plt.xlabel('Actual Power (W)')
    plt.ylabel('Predicted Power (W)')
    plt.title('Actual vs Predicted Power')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('memory_cubic_model.png', dpi=300, bbox_inches='tight')
    
    return {'alpha': alpha, 'beta': beta, 'tau': tau, 'gamma': gamma, 'rmse': rmse, 'r2': r2, 'y_pred': y_pred}

def build_memory_polynomial_model(data):
    """
    Build the memory-aware polynomial model using feature interactions
    """
    print("\n3.3.2 Building Memory-Aware Polynomial Model")
    
    # Prepare features
    features = ['AvgGPUUtil', 'AvgSMClock', 'AvgMemUtil']
    X = data[features].values
    y = data['AvgPower'].values
    
    # Create polynomial features (degree=2 captures all feature pairs)
    poly = PolynomialFeatures(degree=2, include_bias=False)
    X_poly = poly.fit_transform(X)
    
    # Get feature names for the polynomial terms
    #feature_names = poly.get_feature_names_out(features)
    if hasattr(poly, 'get_feature_names_out'):
    	feature_names = poly.get_feature_names_out(features)
    else:
    	feature_names = poly.get_feature_names(features)
        
    print(f"Polynomial features: {feature_names}")
    
    # Fit linear model
    model = LinearRegression()
    model.fit(X_poly, y)
    
    # Get coefficients
    coefs = {name: coef for name, coef in zip(feature_names, model.coef_)}
    gamma = model.intercept_
    
    # Output top coefficients
    print("Top coefficients:")
    for name, coef in sorted(coefs.items(), key=lambda x: abs(x[1]), reverse=True):
        print(f"  {name}: {coef:.6f}")
    print(f"Intercept (γ): {gamma:.4f}")
    
    # Calculate predictions and metrics
    y_pred = model.predict(X_poly)
    rmse = np.sqrt(mean_squared_error(y, y_pred))
    r2 = r2_score(y, y_pred)
    
    print(f"Memory-Aware Polynomial Model RMSE: {rmse:.4f} W, R²: {r2:.4f}")
    
    # Visualize actual vs predicted
    plt.figure(figsize=(10, 6))
    plt.scatter(y, y_pred, alpha=0.6)
    
    # Plot the ideal line (y=x)
    min_val = min(y.min(), y_pred.min())
    max_val = max(y.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='Ideal')
    
    plt.xlabel('Actual Power (W)')
    plt.ylabel('Predicted Power (W)')
    plt.title('Memory-Aware Polynomial Model: Actual vs Predicted Power')
    plt.legend()
    plt.grid(True)
    plt.savefig('memory_polynomial_model.png', dpi=300, bbox_inches='tight')
    
    return {'coefs': coefs, 'gamma': gamma, 'rmse': rmse, 'r2': r2, 'y_pred': y_pred, 'feature_names': feature_names}

def compare_models(models):
    """Compare all models based on their RMSE and R² values"""
    print("\nModel Comparison:")
    
    model_names = list(models.keys())
    rmse_values = [models[name]['rmse'] for name in model_names]
    r2_values = [models[name]['r2'] for name in model_names]
    
    # Create comparison plot
    plt.figure(figsize=(12, 10))
    
    # RMSE comparison
    plt.subplot(2, 1, 1)
    bars = plt.bar(model_names, rmse_values, color='skyblue')
    plt.axhline(y=min(rmse_values), color='red', linestyle='--', alpha=0.7, label='Best RMSE')
    plt.ylabel('RMSE (W)')
    plt.title('Model Comparison: RMSE (lower is better)')
    plt.xticks(rotation=45, ha='right')
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}', ha='center', va='bottom')
    
    # R² comparison
    plt.subplot(2, 1, 2)
    bars = plt.bar(model_names, r2_values, color='lightgreen')
    plt.axhline(y=max(r2_values), color='red', linestyle='--', alpha=0.7, label='Best R²')
    plt.ylabel('R²')
    plt.title('Model Comparison: R² (higher is better)')
    plt.xticks(rotation=45, ha='right')
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.4f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('model_comparison.png', dpi=300, bbox_inches='tight')
    
    # Determine the best model
    best_rmse_model = model_names[rmse_values.index(min(rmse_values))]
    best_r2_model = model_names[r2_values.index(max(r2_values))]
    
    print(f"Best model based on RMSE: {best_rmse_model} (RMSE = {min(rmse_values):.4f})")
    print(f"Best model based on R²: {best_r2_model} (R² = {max(r2_values):.4f})")
    
    return {'best_rmse_model': best_rmse_model, 'best_r2_model': best_r2_model}

def main():
    """Main function to build and compare all power models"""
    # Parse command line arguments
    if len(sys.argv) != 2:
        print("Usage: python build_power_models.py <data_csv_file>")
        sys.exit(1)
    
    data_file = sys.argv[1]
    
    # Load the data
    data = load_data(data_file)
    
    # Build all power models
    models = {}
    
    # 3.1 Basic Power Model
    models['basic'] = build_basic_power_model(data)
    
    # 3.2.1 DVFS-Aware Linear Model
    models['dvfs_linear'] = build_dvfs_linear_model(data)
    
    # 3.2.2 DVFS-Aware Cubic Model
    models['dvfs_cubic'] = build_dvfs_cubic_model(data)
    
    # 3.2.3 DVFS-Aware Interaction Model
    models['dvfs_interaction'] = build_dvfs_interaction_model(data)
    
    # 3.3.1 Memory-Aware Cubic Model
    models['memory_cubic'] = build_memory_cubic_model(data)
    
    # 3.3.2 Memory-Aware Polynomial Model
    models['memory_polynomial'] = build_memory_polynomial_model(data)
    
    # Compare all models
    compare_results = compare_models(models)
    
    # Generate a summary report
    with open('power_model_summary.txt', 'w') as f:
        f.write("===== GPU POWER MODEL SUMMARY =====\n\n")
        
        # Add system information if available
        if os.path.exists('system_info.txt'):
            with open('system_info.txt', 'r') as sysf:
                f.write(sysf.read())
                f.write("\n\n")
        
        f.write("1. Basic Power Model\n")
        f.write("-----------------------\n")
        f.write(f"P = {models['basic']['alpha']:.4f} × utilization + {models['basic']['beta']:.4f}\n")
        f.write(f"RMSE: {models['basic']['rmse']:.4f} W, R²: {models['basic']['r2']:.4f}\n\n")
        
        f.write("2. DVFS-Aware Linear Model\n")
        f.write("--------------------------\n")
        f.write(f"P = {models['dvfs_linear']['alpha']:.4f} × utilization + {models['dvfs_linear']['beta']:.6f} × f + {models['dvfs_linear']['gamma']:.4f}\n")
        f.write(f"RMSE: {models['dvfs_linear']['rmse']:.4f} W, R²: {models['dvfs_linear']['r2']:.4f}\n\n")
        
        f.write("3. DVFS-Aware Cubic Model\n")
        f.write("------------------------\n")
        f.write(f"P = {models['dvfs_cubic']['alpha']:.4f} × utilization + {models['dvfs_cubic']['beta']:.4f} × (f/fmax)³ + {models['dvfs_cubic']['gamma']:.4f}\n")
        f.write(f"fmax = {models['dvfs_cubic']['f_max']:.0f} MHz\n")
        f.write(f"RMSE: {models['dvfs_cubic']['rmse']:.4f} W, R²: {models['dvfs_cubic']['r2']:.4f}\n\n")
        f.write("4. DVFS-Aware Interaction Model\n")
        f.write("------------------------------\n")
        f.write(f"P = {models['dvfs_interaction']['alpha']:.4f} × util + {models['dvfs_interaction']['beta']:.6f} × f + {models['dvfs_interaction']['delta']:.8f} × (util × f) + {models['dvfs_interaction']['gamma']:.4f}\n")
        f.write(f"RMSE: {models['dvfs_interaction']['rmse']:.4f} W, R²: {models['dvfs_interaction']['r2']:.4f}\n\n")
        
        f.write("5. Memory-Aware Cubic Model\n")
        f.write("--------------------------\n")
        f.write(f"P = {models['memory_cubic']['alpha']:.4f} × util + {models['memory_cubic']['beta']:.8f} × f³ × M + {models['memory_cubic']['tau']:.6f} × f + {models['memory_cubic']['gamma']:.4f}\n")
        f.write(f"where M = 1 + 0.5 × (memory_utilization)\n")
        f.write(f"RMSE: {models['memory_cubic']['rmse']:.4f} W, R²: {models['memory_cubic']['r2']:.4f}\n\n")
        
        f.write("6. Memory-Aware Polynomial Model\n")
        f.write("-------------------------------\n")
        f.write("P = Σ(wi × (featurei) × (featurej)) + γ\n")
        f.write("Top coefficients:\n")
        for name, coef in sorted(models['memory_polynomial']['coefs'].items(), key=lambda x: abs(x[1]), reverse=True)[:5]:
            f.write(f"  {name}: {coef:.6f}\n")
        f.write(f"Intercept (γ): {models['memory_polynomial']['gamma']:.4f}\n")
        f.write(f"RMSE: {models['memory_polynomial']['rmse']:.4f} W, R²: {models['memory_polynomial']['r2']:.4f}\n\n")
        
        f.write("===== MODEL COMPARISON =====\n\n")
        f.write(f"Best model based on RMSE: {compare_results['best_rmse_model']} (RMSE = {models[compare_results['best_rmse_model']]['rmse']:.4f})\n")
        f.write(f"Best model based on R²: {compare_results['best_r2_model']} (R² = {models[compare_results['best_r2_model']]['r2']:.4f})\n")
        
if __name__ == "__main__":
    main()
PYTHON_SCRIPT

# Make the Python script executable
chmod +x "${OUTPUT_DIR}/build_power_models.py"

# Run the model building script
echo "Building power models..."
cd $OUTPUT_DIR
python3 ./build_power_models.py histogram_data_summary.csv > power_model_log.txt 2>&1

# Check if the script ran successfully
if [ $? -eq 0 ]; then
    echo "Power models built successfully!"
    echo "Output files:"
    ls -la *.png
    echo ""
    
    # Display the model summary
    if [ -f power_model_summary.txt ]; then
        echo "===== POWER MODEL SUMMARY ====="
        cat power_model_summary.txt
    fi
else
    echo "Error building power models. Check ${OUTPUT_DIR}/power_model_log.txt for details."
fi

# Create a single PDF report with all visualizations
if command -v convert > /dev/null; then
    echo "Creating final report PDF..."
    convert system_info.txt *.png power_model_summary.txt histogram_power_model_report.pdf
    echo "Report saved as ${OUTPUT_DIR}/histogram_power_model_report.pdf"
fi

echo ""
echo "===== HISTOGRAM POWER MODEL ANALYSIS COMPLETE ====="
echo "Results saved to: ${OUTPUT_DIR}"
echo ""
echo "Summary of collected data is in: histogram_data_summary.csv"
echo "Power model visualizations are saved as PNG files"
echo "Complete power model details are in: power_model_summary.txt"
