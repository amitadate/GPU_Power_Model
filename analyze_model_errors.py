#!/usr/bin/env python3


import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime 

def find_most_recent_file(directory, pattern):
    directory = Path(directory)
    matching_files = list(directory.glob(pattern))
    if not matching_files:
        return None
    return max(matching_files, key=lambda p: p.stat().st_mtime)

def load_data():
    # Find the most recent enhanced data file
    enhanced_data_file = find_most_recent_file("power_results", "enhanced_histogram_power_data_*.csv")
    if not enhanced_data_file:
        print("No enhanced power data file found. Please run the power predictor first.")
        sys.exit(1)

    print(f"Loading data from {enhanced_data_file}")
    df = pd.read_csv(enhanced_data_file)
    
    # Verify that it contains model predictions
    if 'basic_prediction' not in df.columns or 'dvfs_prediction' not in df.columns:
        print("The data file does not contain model predictions.")
        sys.exit(1)
    
    # load model stats
    stats_file = find_most_recent_file("power_results", "model_prediction_stats_*.csv")
    stats_df = None
    if stats_file:
        print(f"Loading model stats from {stats_file}")
        stats_df = pd.read_csv(stats_file)
    
    return df, stats_df

def analyze_overall_errors(stats_df):
    if stats_df is None or stats_df.empty:
        print("No model stats available for analysis.")
        return
    
    # Extract error metrics for each model
    print("\n===== OVERALL MODEL ERROR METRICS =====")
    print(f"Number of samples: {stats_df['samples'].values[0]}")
    
    # Create comparison table
    if 'basic_mae' in stats_df.columns and 'dvfs_mae' in stats_df.columns:
        print("\nError Metrics Comparison:")
        comparison = pd.DataFrame({
            'Metric': ['Mean Absolute Error (W)', 'RMSE (W)', 'Error Percentage (%)'],
            'Basic Model': [
                stats_df['basic_mae'].values[0],
                stats_df['basic_rmse'].values[0],
                stats_df['basic_error_pct'].values[0]
            ],
            'DVFS Model': [
                stats_df['dvfs_mae'].values[0],
                stats_df['dvfs_rmse'].values[0],
                stats_df['dvfs_error_pct'].values[0]
            ]
        })
        print(comparison.to_string(index=False))
        
        # Calculate improvement
        mae_improvement = ((stats_df['basic_mae'].values[0] - stats_df['dvfs_mae'].values[0]) / 
                          stats_df['basic_mae'].values[0] * 100)
        rmse_improvement = ((stats_df['basic_rmse'].values[0] - stats_df['dvfs_rmse'].values[0]) / 
                           stats_df['basic_rmse'].values[0] * 100)
        
        print(f"\nDVFS Model Improvement:")
        print(f"  MAE Improvement: {mae_improvement:.2f}%")
        print(f"  RMSE Improvement: {rmse_improvement:.2f}%")

def analyze_error_by_power_state(df):
    """Analyze errors based on different GPU power states."""
    # Define power states based on clock frequency
    df['power_state'] = 'Unknown'
    df.loc[df['sm_clock'] < 400, 'power_state'] = 'Idle'
    df.loc[(df['sm_clock'] >= 400) & (df['sm_clock'] < 1000), 'power_state'] = 'Medium'
    df.loc[df['sm_clock'] >= 1000, 'power_state'] = 'High'
    
    # Calculate errors by power state
    state_errors = df.groupby('power_state').agg({
        'power_watts': 'mean',
        'basic_abs_error': 'mean',
        'dvfs_abs_error': 'mean',
        'basic_error': 'mean',  
        'dvfs_error': 'mean',
        'sm_clock': 'mean'
    }).reset_index()
    
    state_errors['basic_error_pct'] = (state_errors['basic_abs_error'] / state_errors['power_watts'] * 100)
    state_errors['dvfs_error_pct'] = (state_errors['dvfs_abs_error'] / state_errors['power_watts'] * 100)
    
    # Calculate sample counts
    state_counts = df['power_state'].value_counts().reset_index()
    state_counts.columns = ['power_state', 'sample_count']
    state_errors = pd.merge(state_errors, state_counts, on='power_state')
    
    # Sort by frequency
    state_errors = state_errors.sort_values('sample_count', ascending=False)
    
    print("\n===== ERRORS BY POWER STATE =====")
    
    # Create a readable output
    report = pd.DataFrame({
        'Power State': state_errors['power_state'],
        'Avg Clock (MHz)': state_errors['sm_clock'].round(0),
        'Avg Power (W)': state_errors['power_watts'].round(2),
        'Samples': state_errors['sample_count'],
        'Basic MAE (W)': state_errors['basic_abs_error'].round(2),
        'DVFS MAE (W)': state_errors['dvfs_abs_error'].round(2),
        'Basic Error (%)': state_errors['basic_error_pct'].round(2),
        'DVFS Error (%)': state_errors['dvfs_error_pct'].round(2),
        'Basic Bias': state_errors['basic_error'].round(2),
        'DVFS Bias': state_errors['dvfs_error'].round(2)
    })
    
    print(report.to_string(index=False))
    
    # Create a bar chart 
    plt.figure(figsize=(12, 6))
    
    # Set up bar positions
    states = state_errors['power_state']
    x = np.arange(len(states))
    width = 0.35
    
    # Create bars
    plt.bar(x - width/2, state_errors['basic_error_pct'], width, label='Basic Model Error (%)')
    plt.bar(x + width/2, state_errors['dvfs_error_pct'], width, label='DVFS Model Error (%)')
    
    # Add labels and formatting
    plt.xlabel('GPU Power State')
    plt.ylabel('Error Percentage (%)')
    plt.title('Model Error by GPU Power State')
    plt.xticks(x, states)
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Add value labels on bars
    for i, v in enumerate(state_errors['basic_error_pct']):
        plt.text(i - width/2, v + 0.5, f"{v:.1f}%", ha='center')
    
    for i, v in enumerate(state_errors['dvfs_error_pct']):
        plt.text(i + width/2, v + 0.5, f"{v:.1f}%", ha='center')
    
    # Save and show the plot
    plt.tight_layout()
    plt.savefig('power_plots/error_by_power_state.png')
    
    return state_errors

def analyze_errors_over_time(df):
    plt.figure(figsize=(15, 10))
    
    # Plot the actual power and predictions
    plt.subplot(2, 1, 1)
    plt.plot(df['timestamp'], df['power_watts'], 'k-', label='Actual Power')
    plt.plot(df['timestamp'], df['basic_prediction'], 'b--', label='Basic Model')
    plt.plot(df['timestamp'], df['dvfs_prediction'], 'r--', label='DVFS Model')
    plt.xlabel('Time (s)')
    plt.ylabel('Power (W)')
    plt.title('Actual vs. Predicted Power During Histogram Kernel Execution')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Plot the absolute errors
    plt.subplot(2, 1, 2)
    plt.plot(df['timestamp'], df['basic_abs_error'], 'b-', label='Basic Model Error')
    plt.plot(df['timestamp'], df['dvfs_abs_error'], 'r-', label='DVFS Model Error')
    plt.xlabel('Time (s)')
    plt.ylabel('Absolute Error (W)')
    plt.title('Model Prediction Errors Over Time')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Add a secondary axis for clock frequency
    ax2 = plt.twinx()
    ax2.plot(df['timestamp'], df['sm_clock'], 'g:', alpha=0.6, label='SM Clock')
    ax2.set_ylabel('SM Clock (MHz)', color='g')
    ax2.tick_params(axis='y', labelcolor='g')
    
    plt.tight_layout()
    plt.savefig('power_plots/errors_over_time.png')
    
    # Calculate average errors during each phase (idle, active, transitioning)
    kernel_runs = df[df['kernel'] != 'None']
    idle_periods = df[df['kernel'] == 'None']
    
    # Count transitions (where clock changes significantly)
    transitions = df[(df['sm_clock'].diff().abs() > 100) & (df['sm_clock'] > 400)]
    
    # Calculate stats for each phase
    print("\n===== ERRORS BY EXECUTION PHASE =====")
    phases = {
        'Idle': idle_periods,
        'Active Execution': kernel_runs,
        'Transitions': transitions
    }
    
    phase_stats = []
    for phase_name, phase_data in phases.items():
        if len(phase_data) == 0:
            continue
            
        stats = {
            'Phase': phase_name,
            'Samples': len(phase_data),
            'Avg Power (W)': phase_data['power_watts'].mean(),
            'Basic MAE (W)': phase_data['basic_abs_error'].mean(),
            'DVFS MAE (W)': phase_data['dvfs_abs_error'].mean(),
            'Basic Error (%)': (phase_data['basic_abs_error'] / phase_data['power_watts'] * 100).mean(),
            'DVFS Error (%)': (phase_data['dvfs_abs_error'] / phase_data['power_watts'] * 100).mean()
        }
        phase_stats.append(stats)
    
    phase_df = pd.DataFrame(phase_stats)
    print(phase_df.to_string(index=False))
    
    return phase_df

def analyze_correlation_with_features(df):
    # Calculate correlations between errors and features
    features = ['power_watts', 'utilization', 'sm_clock', 'mem_clock', 'temperature']
    error_cols = ['basic_abs_error', 'dvfs_abs_error']
    
    # Filter to only include relevant columns that exist
    features = [f for f in features if f in df.columns]
    error_cols = [e for e in error_cols if e in df.columns]
    
    if not features or not error_cols:
        print("Missing required columns for correlation analysis")
        return
    
    corr_columns = features + error_cols
    corr = df[corr_columns].corr()
    
    # Print correlation with errors
    print("\n===== ERROR CORRELATION WITH FEATURES =====")
    for error in error_cols:
        print(f"\nCorrelation with {error}:")
        for feature in features:
            correlation = corr.loc[error, feature]
            print(f"  {feature}: {correlation:.4f}")
    
    # Create scatter plots of errors vs. key features
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Clock frequency vs. errors
    if 'sm_clock' in features:
        axes[0, 0].scatter(df['sm_clock'], df['basic_abs_error'], alpha=0.5, label='Basic Model')
        axes[0, 0].scatter(df['sm_clock'], df['dvfs_abs_error'], alpha=0.5, label='DVFS Model')
        axes[0, 0].set_xlabel('SM Clock (MHz)')
        axes[0, 0].set_ylabel('Absolute Error (W)')
        axes[0, 0].set_title('Errors vs. Clock Frequency')
        axes[0, 0].legend()
        axes[0, 0].grid(True, linestyle='--', alpha=0.7)
    
    # Utilization vs. errors
    if 'utilization' in features:
        axes[0, 1].scatter(df['utilization'], df['basic_abs_error'], alpha=0.5, label='Basic Model')
        axes[0, 1].scatter(df['utilization'], df['dvfs_abs_error'], alpha=0.5, label='DVFS Model')
        axes[0, 1].set_xlabel('Utilization (%)')
        axes[0, 1].set_ylabel('Absolute Error (W)')
        axes[0, 1].set_title('Errors vs. Utilization')
        axes[0, 1].legend()
        axes[0, 1].grid(True, linestyle='--', alpha=0.7)
    
    # Actual power vs. errors
    axes[1, 0].scatter(df['power_watts'], df['basic_abs_error'], alpha=0.5, label='Basic Model')
    axes[1, 0].scatter(df['power_watts'], df['dvfs_abs_error'], alpha=0.5, label='DVFS Model')
    axes[1, 0].set_xlabel('Actual Power (W)')
    axes[1, 0].set_ylabel('Absolute Error (W)')
    axes[1, 0].set_title('Errors vs. Actual Power')
    axes[1, 0].legend()
    axes[1, 0].grid(True, linestyle='--', alpha=0.7)
    
    # Error distributions
    axes[1, 1].hist(df['basic_error'], bins=30, alpha=0.5, label='Basic Model')
    axes[1, 1].hist(df['dvfs_error'], bins=30, alpha=0.5, label='DVFS Model')
    axes[1, 1].set_xlabel('Error (W)')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].set_title('Error Distributions')
    axes[1, 1].legend()
    axes[1, 1].grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig('power_plots/error_correlations.png')

def analyze_error_distribution(df):
    plt.figure(figsize=(12, 8))
    
    # Basic Model Error Distribution
    plt.subplot(2, 1, 1)
    plt.hist(df['basic_error'], bins=30, alpha=0.7)
    plt.axvline(x=0, color='r', linestyle='--')
    plt.xlabel('Error (W)')
    plt.ylabel('Frequency')
    plt.title('Basic Model Error Distribution')
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # DVFS Model Error Distribution
    plt.subplot(2, 1, 2)
    plt.hist(df['dvfs_error'], bins=30, alpha=0.7)
    plt.axvline(x=0, color='r', linestyle='--')
    plt.xlabel('Error (W)')
    plt.ylabel('Frequency')
    plt.title('DVFS Model Error Distribution')
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig('power_plots/error_distributions.png')
    
    # Calculate statistics on error distributions
    basic_errors = df['basic_error']
    dvfs_errors = df['dvfs_error']
    
    print("\n===== ERROR DISTRIBUTION STATISTICS =====")
    print("Basic Model Error Distribution:")
    print(f"  Mean: {basic_errors.mean():.2f} W")
    print(f"  Std Dev: {basic_errors.std():.2f} W")
    print(f"  Skewness: {basic_errors.skew():.4f}")
    print(f"  % Positive Errors (Under-prediction): {(basic_errors > 0).mean() * 100:.1f}%")
    print(f"  % Negative Errors (Over-prediction): {(basic_errors < 0).mean() * 100:.1f}%")
    
    print("\nDVFS Model Error Distribution:")
    print(f"  Mean: {dvfs_errors.mean():.2f} W")
    print(f"  Std Dev: {dvfs_errors.std():.2f} W")
    print(f"  Skewness: {dvfs_errors.skew():.4f}")
    print(f"  % Positive Errors (Under-prediction): {(dvfs_errors > 0).mean() * 100:.1f}%")
    print(f"  % Negative Errors (Over-prediction): {(dvfs_errors < 0).mean() * 100:.1f}%")


def create_combined_error_plots(df):
    """Create combined error analysis plots for all models."""
    plt.figure(figsize=(15, 10))
    
    # Plot errors by power level
    plt.subplot(2, 2, 1)
    power_bins = [0, 20, 40, 60, 80, 100, 120]
    df['power_bin'] = pd.cut(df['power_watts'], power_bins)
    
    power_errors = df.groupby('power_bin').agg({
        'basic_abs_error': 'mean',
        'dvfs_abs_error': 'mean'
    }).reset_index()
    
    if 'hybrid_abs_error' in df.columns:
        power_errors['hybrid_abs_error'] = df.groupby('power_bin')['hybrid_abs_error'].mean().values
    
    bin_centers = [(power_bins[i] + power_bins[i+1])/2 for i in range(len(power_bins)-1)]
    
    plt.plot(bin_centers, power_errors['basic_abs_error'], 'bo-', label='Basic Model')
    plt.plot(bin_centers, power_errors['dvfs_abs_error'], 'ro-', label='DVFS Model')
    
    if 'hybrid_abs_error' in df.columns:
        plt.plot(bin_centers, power_errors['hybrid_abs_error'], 'go-', label='Hybrid Model')
    
    plt.xlabel('Power Level (W)')
    plt.ylabel('Mean Absolute Error (W)')
    plt.title('Model Error by Power Level')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Plot errors by clock frequency
    plt.subplot(2, 2, 2)
    clock_bins = [0, 400, 800, 1200, 1600, 2000]
    df['clock_bin'] = pd.cut(df['sm_clock'], clock_bins)
    
    clock_errors = df.groupby('clock_bin').agg({
        'basic_abs_error': 'mean',
        'dvfs_abs_error': 'mean'
    }).reset_index()
    
    if 'hybrid_abs_error' in df.columns:
        clock_errors['hybrid_abs_error'] = df.groupby('clock_bin')['hybrid_abs_error'].mean().values
    
    bin_centers = [(clock_bins[i] + clock_bins[i+1])/2 for i in range(len(clock_bins)-1)]
    
    plt.plot(bin_centers, clock_errors['basic_abs_error'], 'bo-', label='Basic Model')
    plt.plot(bin_centers, clock_errors['dvfs_abs_error'], 'ro-', label='DVFS Model')
    
    if 'hybrid_abs_error' in df.columns:
        plt.plot(bin_centers, clock_errors['hybrid_abs_error'], 'go-', label='Hybrid Model')
    
    plt.xlabel('Clock Frequency (MHz)')
    plt.ylabel('Mean Absolute Error (W)')
    plt.title('Model Error by Clock Frequency')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Plot errors by utilization
    plt.subplot(2, 2, 3)
    util_bins = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    df['util_bin'] = pd.cut(df['utilization'], util_bins)
    
    util_errors = df.groupby('util_bin').agg({
        'basic_abs_error': 'mean',
        'dvfs_abs_error': 'mean'
    }).reset_index()
    
    if 'hybrid_abs_error' in df.columns:
        util_errors['hybrid_abs_error'] = df.groupby('util_bin')['hybrid_abs_error'].mean().values
    
    bin_centers = [(util_bins[i] + util_bins[i+1])/2 for i in range(len(util_bins)-1)]
    
    plt.plot(bin_centers, util_errors['basic_abs_error'], 'bo-', label='Basic Model')
    plt.plot(bin_centers, util_errors['dvfs_abs_error'], 'ro-', label='DVFS Model')
    
    if 'hybrid_abs_error' in df.columns:
        plt.plot(bin_centers, util_errors['hybrid_abs_error'], 'go-', label='Hybrid Model')
    
    plt.xlabel('Utilization (%)')
    plt.ylabel('Mean Absolute Error (W)')
    plt.title('Model Error by Utilization')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Plot model prediction vs actual power
    plt.subplot(2, 2, 4)
    plt.scatter(df['power_watts'], df['basic_prediction'], alpha=0.5, label='Basic Model')
    plt.scatter(df['power_watts'], df['dvfs_prediction'], alpha=0.5, label='DVFS Model')
    
    if 'hybrid_prediction' in df.columns:
        plt.scatter(df['power_watts'], df['hybrid_prediction'], alpha=0.5, label='Hybrid Model')
    
    # Add perfect prediction line
    max_power = max(df['power_watts'].max(), df['basic_prediction'].max(), df['dvfs_prediction'].max())
    plt.plot([0, max_power], [0, max_power], 'k--')
    
    plt.xlabel('Actual Power (W)')
    plt.ylabel('Predicted Power (W)')
    plt.title('Model Predictions vs Actual Power')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    combined_plot = f"power_plots/combined_error_analysis_{timestamp}.png"
    plt.savefig(combined_plot)
    plt.close()
    
    return combined_plot

def main():
    # Create power_plots directory
    Path("power_plots").mkdir(exist_ok=True)
    
    # Load data
    df, stats_df = load_data()
    
    # Print info
    print(f"Data loaded: {len(df)} samples")
    print(f"Kernel: {df['kernel'].unique()}")
    print(f"Power range: {df['power_watts'].min():.2f}W - {df['power_watts'].max():.2f}W")
    print(f"Clock range: {df['sm_clock'].min():.0f}MHz - {df['sm_clock'].max():.0f}MHz")
    
    # Run analyses
    analyze_overall_errors(stats_df)
    state_errors = analyze_error_by_power_state(df)
    phase_df = analyze_errors_over_time(df)
    analyze_correlation_with_features(df)
    analyze_error_distribution(df)

    combined_plot = create_combined_error_plots(df)
    print(f"Combined error analysis saved to {combined_plot}")
    
    print("\nAnalysis complete! Visualizations saved to power_plots directory.")

if __name__ == "__main__":
    main()
