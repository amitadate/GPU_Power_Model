#!/usr/bin/env python3

import os
import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime


sys.path.insert(0, str(Path(__file__).resolve().parent))


from file_utils import find_most_recent_power_data, find_most_recent_power_stats, create_timestamp, ensure_directory

def calculate_energy_metrics(power_data, execution_times):
    # Group power data by kernel
    kernel_power = power_data.groupby('kernel')['power_watts'].mean().reset_index()
    
    # Merge with execution times
    if isinstance(execution_times, dict):
        # Convert dictionary to DataFrame
        exec_df = pd.DataFrame([
            {'kernel': k, 'execution_time_sec': v} 
            for k, v in execution_times.items()
        ])
    else:
        exec_df = execution_times
    
    # Merge the dataframes
    metrics_df = pd.merge(kernel_power, exec_df, on='kernel', how='inner')
    
    # Calculate energy consumption (joules = watts * seconds)
    metrics_df['energy_joules'] = metrics_df['power_watts'] * metrics_df['execution_time_sec']
    
    # Calculate Energy-Delay Product (EDP)
    metrics_df['edp'] = metrics_df['energy_joules'] * metrics_df['execution_time_sec']
    
    # Calculate Energy-Delay^2 Product (ED2P) - emphasizes performance more
    metrics_df['ed2p'] = metrics_df['energy_joules'] * (metrics_df['execution_time_sec'] ** 2)
    
    # Normalize metrics for comparison
    min_energy = metrics_df['energy_joules'].min()
    min_edp = metrics_df['edp'].min()
    min_ed2p = metrics_df['ed2p'].min()
    
    metrics_df['energy_normalized'] = metrics_df['energy_joules'] / min_energy
    metrics_df['edp_normalized'] = metrics_df['edp'] / min_edp
    metrics_df['ed2p_normalized'] = metrics_df['ed2p'] / min_ed2p
    
    # Calculate performance per watt (higher is better)
    metrics_df['performance_per_watt'] = 1 / metrics_df['energy_joules']
    
    # Normalize performance per watt
    max_perf_per_watt = metrics_df['performance_per_watt'].max()
    metrics_df['performance_per_watt_normalized'] = metrics_df['performance_per_watt'] / max_perf_per_watt
    
    return metrics_df

def create_energy_visualizations(metrics_df, output_dir):
    output_path = ensure_directory(output_dir)
    
    # Timestamp for filenames
    timestamp = create_timestamp()
    
    # Bar colors for different kernels
    colors = plt.cm.viridis(np.linspace(0, 1, len(metrics_df)))
    
    # Create energy consumption plot
    plt.figure(figsize=(12, 6))
    
    # Sort by energy consumption (ascending)
    sorted_df = metrics_df.sort_values('energy_joules')
    
    plt.bar(sorted_df['kernel'], sorted_df['energy_joules'], color=colors)
    plt.xlabel("Kernel Implementation")
    plt.ylabel("Energy Consumption (Joules)")
    plt.title("Energy Consumption by Kernel Implementation")
    plt.xticks(rotation=45)
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)
    
    # Add labels with execution time
    for i, row in enumerate(sorted_df.itertuples()):
        plt.text(i, row.energy_joules + 1, f"{row.execution_time_sec:.4f}s", 
                 ha='center', va='bottom', rotation=0,
                 bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.2'))
    
    plt.tight_layout()
    
    energy_plot = output_path / f"energy_consumption_{timestamp}.png"
    plt.savefig(energy_plot)
    plt.close()
    
    # Create multi-metric comparison plot
    plt.figure(figsize=(14, 10))
    
    metrics = ['energy_normalized', 'edp_normalized', 'ed2p_normalized']
    metric_labels = ['Normalized Energy', 'Normalized EDP', 'Normalized ED²P']
    
    # Sort by EDP (ascending)
    sorted_df = metrics_df.sort_values('edp')
    
    for i, (metric, label) in enumerate(zip(metrics, metric_labels)):
        plt.subplot(3, 1, i+1)
        plt.bar(sorted_df['kernel'], sorted_df[metric], color=colors)
        plt.ylabel(label)
        plt.title(f"{label} by Kernel Implementation (lower is better)")
        plt.xticks(rotation=45)
        plt.grid(True, axis='y', linestyle='--', alpha=0.7)
        
        # Add labels
        for j, row in enumerate(sorted_df.itertuples()):
            value = getattr(row, metric)
            plt.text(j, value + 0.05, f"{value:.2f}x", 
                     ha='center', va='bottom', rotation=0,
                     bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.2'))
    
    plt.tight_layout()
    
    metrics_plot = output_path / f"energy_metrics_{timestamp}.png"
    plt.savefig(metrics_plot)
    plt.close()
    
    # Create Pareto frontier plot (execution time vs. energy)
    plt.figure(figsize=(10, 8))
    
    # Scatter plot of execution time vs. energy
    plt.scatter(metrics_df['execution_time_sec'], metrics_df['energy_joules'], 
                s=100, c=range(len(metrics_df)), cmap='viridis')
    
    # Add kernel labels
    for i, row in enumerate(metrics_df.itertuples()):
        plt.annotate(row.kernel, 
                     (row.execution_time_sec, row.energy_joules),
                     xytext=(5, 5), textcoords='offset points')
    
    # Draw Pareto frontier
    # Sort by execution time
    pareto_df = metrics_df.sort_values('execution_time_sec').copy()
    pareto_df['pareto'] = False
    
    # Find Pareto points
    min_energy = float('inf')
    for i in reversed(range(len(pareto_df))):
        if pareto_df.iloc[i]['energy_joules'] < min_energy:
            min_energy = pareto_df.iloc[i]['energy_joules']
            pareto_df.iloc[i, pareto_df.columns.get_loc('pareto')] = True
    
    # Connect Pareto points
    pareto_points = pareto_df[pareto_df['pareto']].sort_values('execution_time_sec')
    plt.plot(pareto_points['execution_time_sec'], pareto_points['energy_joules'], 
             'r--', linewidth=2, label='Pareto Frontier')
    
    plt.xlabel('Execution Time (seconds)')
    plt.ylabel('Energy Consumption (Joules)')
    plt.title('Pareto Frontier: Execution Time vs. Energy Consumption')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    
    pareto_plot = output_path / f"pareto_frontier_{timestamp}.png"
    plt.savefig(pareto_plot)
    plt.close()
    
    print(f"Energy visualizations saved to {output_path}:")
    print(f"  - {energy_plot.name}")
    print(f"  - {metrics_plot.name}")
    print(f"  - {pareto_plot.name}")
    
    return {
        'energy_plot': energy_plot,
        'metrics_plot': metrics_plot,
        'pareto_plot': pareto_plot
    }

def extract_execution_times_from_stats(stats_df):
    if 'execution_time_sec' in stats_df.columns:
        return dict(zip(stats_df['kernel'], stats_df['execution_time_sec']))
    else:
        print("No execution time information found in stats DataFrame")
        return None

def infer_execution_times_from_power_data(power_data):

    kernel_times = {}
    for kernel in power_data['kernel'].unique():
        if kernel == 'None':
            continue
            
        kernel_df = power_data[power_data['kernel'] == kernel]
        if not kernel_df.empty:
            # Use the difference between max and min timestamp
            kernel_times[kernel] = kernel_df['timestamp'].max() - kernel_df['timestamp'].min()
    
    if not kernel_times:
        print("Could not infer execution times from power data")
        return None
    
    return kernel_times

def main():
    # Find the most recent files
    power_data_path = find_most_recent_power_data()
    power_stats_path = find_most_recent_power_stats()
    
    if power_data_path is None:
        print("Error: No power data CSV files found in power_results directory")
        print("Please run histogram_power_harness.py first")
        return
    
    print(f"Automatically selected most recent power data file: {power_data_path}")
    
    if power_stats_path:
        print(f"Automatically selected most recent power stats file: {power_stats_path}")
    else:
        print("No power stats file found. Will try to infer execution times from power data.")
    
    # Load the power data
    try:
        power_data = pd.read_csv(power_data_path)
        print(f"Loaded power data from {power_data_path}")
        print(f"Data contains {len(power_data)} samples across {power_data['kernel'].nunique()} kernels")
    except Exception as e:
        print(f"Error loading power data: {e}")
        return
    
    # Get execution times
    execution_times = None
    
    if power_stats_path:
        try:
            stats_df = pd.read_csv(power_stats_path)
            execution_times = extract_execution_times_from_stats(stats_df)
        except Exception as e:
            print(f"Error reading stats file: {e}")
    
    if execution_times is None:
        # Try to infer from power data
        execution_times = infer_execution_times_from_power_data(power_data)
    
    if execution_times is None:
        print("Could not obtain execution times. Cannot calculate energy metrics.")
        return
    
    print("Using execution times:")
    for kernel, time in execution_times.items():
        print(f"  {kernel}: {time:.6f} seconds")
    
    # Calculate energy metrics
    metrics_df = calculate_energy_metrics(power_data, execution_times)
    
    # Save the metrics
    timestamp = create_timestamp()
    output_dir = ensure_directory("energy_analysis")
    
    metrics_file = output_dir / f"energy_metrics_{timestamp}.csv"
    metrics_df.to_csv(metrics_file, index=False)
    print(f"Energy metrics saved to {metrics_file}")
    
    # Create visualizations
    create_energy_visualizations(metrics_df, output_dir)
    
    # Print summary of results
    print("\nEnergy Efficiency Summary:")
    print(metrics_df[['kernel', 'power_watts', 'execution_time_sec', 'energy_joules', 'edp']].to_string(index=False))
    
    # Identify the most efficient implementation for different metrics
    best_energy = metrics_df.loc[metrics_df['energy_joules'].idxmin()]
    best_edp = metrics_df.loc[metrics_df['edp'].idxmin()]
    best_ed2p = metrics_df.loc[metrics_df['ed2p'].idxmin()]
    
    print("\nMost Energy Efficient Implementation:")
    print(f"  {best_energy['kernel']}: {best_energy['energy_joules']:.2f} Joules")
    
    print("\nBest Energy-Delay Product (EDP):")
    print(f"  {best_edp['kernel']}: {best_edp['edp']:.2f}")
    
    print("\nBest Energy-Delay² Product (ED²P):")
    print(f"  {best_ed2p['kernel']}: {best_ed2p['ed2p']:.2f}")
    
    # Provide recommendations
    print("\nRecommendations:")
    
    # For high energy efficiency
    print(f"  For maximum energy efficiency (lowest total energy): {best_energy['kernel']}")
    
    # For balanced performance and energy
    print(f"  For balanced performance and energy (best EDP): {best_edp['kernel']}")
    
    # For performance-critical applications
    print(f"  For performance-critical applications (best ED²P): {best_ed2p['kernel']}")

if __name__ == "__main__":
    main()
