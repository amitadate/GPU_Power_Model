#!/usr/bin/env python3

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime

# Check for file_utils
try:
    from file_utils import find_most_recent_file, create_timestamp
except ImportError:
    print("Warning: file_utils not found, using built-in functions")
    
    def find_most_recent_file(directory, pattern):
        """Find the most recent file in a directory that matches the pattern."""
        import re
        from pathlib import Path
        directory = Path(directory)
        if not directory.exists():
            return None
        matching_files = [f for f in directory.glob('*') if re.search(pattern, f.name)]
        if not matching_files:
            return None
        return max(matching_files, key=lambda f: f.stat().st_mtime)
    
    def create_timestamp():
        from datetime import datetime
        return datetime.now().strftime("%Y%m%d_%H%M%S")

def ensure_directory(path):
    """Ensure a directory exists."""
    path = Path(path)
    path.mkdir(exist_ok=True, parents=True)
    return path

def find_stats_file():
    """Find the most recent histogram power stats file."""
    stats_file = find_most_recent_file("power_results", r"^histogram_power_stats_\d{8}_\d{6}\.csv$")
    if stats_file is None:
        print("No power stats files found in power_results directory.")
        sys.exit(1)
    return stats_file

def analyze_optimization_techniques(stats_file=None):
    # Find the most recent stats file if not provided
    if stats_file is None:
        stats_file = find_stats_file()
    
    print(f"Analyzing optimization techniques using data from {stats_file}")
    
    # Load the stats data
    df = pd.read_csv(stats_file)
    
    # Check if we have the necessary columns
    required_columns = ['kernel', 'avg_power_watts', 'execution_time_sec']
    if not all(col in df.columns for col in required_columns):
        print(f"Stats file missing required columns. Found: {df.columns}")
        return
    
    # Rename columns for consistency
    df = df.rename(columns={
        'avg_power_watts': 'power_watts',
        'avg_clock_mhz': 'clock_mhz'
    })
    
    # Calculate energy metrics if not already present
    if 'energy_joules' not in df.columns:
        df['energy_joules'] = df['power_watts'] * df['execution_time_sec']
    
    if 'edp' not in df.columns:
        df['edp'] = df['energy_joules'] * df['execution_time_sec']
    
    if 'ed2p' not in df.columns:
        df['ed2p'] = df['edp'] * df['execution_time_sec']
    
    # Filter out rows with NaN values
    df = df.dropna(subset=['energy_joules', 'execution_time_sec'])
    
    if df.empty:
        print("No valid data to analyze after filtering NaN values.")
        return
    
    # Create a mapping of optimizations for each strategy
    optimization_techniques = {
        "Baseline": ["Global Memory Atomics"],
        "Strategy1": ["Private Histograms", "Final Reduction"],
        "Strategy2": ["Shared Memory", "Global Atomics"],
        "Strategy3": ["Shared Memory", "Input Tile Loading"],
        "Strategy4": ["Shared Memory", "Bank Conflict Avoidance", "Padded Indexing"],
        "Strategy5": ["Shared Memory", "Coalesced Access", "Memory Optimization"],
        "Strategy6": ["Optimized Tile Size", "Coalesced Access", "Direct Global Memory"],
        "Strategy7": ["Linear Indexing", "Optimized Shared Memory", "Coalesced Access"],
        "Strategy8": ["Local Histogram", "Improved Memory Layout", "Input Tiling"],
        "Strategy9": ["Per-block Local Histogram", "Efficient Reduction", "Optimized Memory"]
    }
    
    # Add optimization techniques to dataframe
    df['optimizations'] = df['kernel'].map(lambda k: ", ".join(optimization_techniques.get(k, [])))
    
    # Create a directory for output
    output_dir = ensure_directory("energy_analysis")
    timestamp = create_timestamp()
    
    # Save the enhanced data
    enhanced_df_file = output_dir / f"kernel_energy_analysis_{timestamp}.csv"
    df.to_csv(enhanced_df_file, index=False)
    print(f"Enhanced energy data saved to {enhanced_df_file}")
    
    # --- Create visualizations ---
    
    # 1. Energy consumption by kernel
    plt.figure(figsize=(12, 6))
    sorted_df = df.sort_values('energy_joules')
    
    bars = plt.bar(sorted_df['kernel'], sorted_df['energy_joules'], color='skyblue')
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{height:.6f}',
                ha='center', va='bottom', rotation=0)
    
    plt.xlabel("Kernel Implementation")
    plt.ylabel("Energy Consumption (Joules)")
    plt.title("Energy Consumption by Kernel Implementation")
    plt.xticks(rotation=45)
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    energy_plot = output_dir / f"kernel_energy_{timestamp}.png"
    plt.savefig(energy_plot)
    plt.close()
    
    # 2. Execution time by kernel
    plt.figure(figsize=(12, 6))
    time_sorted_df = df.sort_values('execution_time_sec')
    
    bars = plt.bar(time_sorted_df['kernel'], time_sorted_df['execution_time_sec'], color='lightgreen')
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{height:.6f}',
                ha='center', va='bottom', rotation=0)
    
    plt.xlabel("Kernel Implementation")
    plt.ylabel("Execution Time (seconds)")
    plt.title("Execution Time by Kernel Implementation")
    plt.xticks(rotation=45)
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    time_plot = output_dir / f"kernel_execution_time_{timestamp}.png"
    plt.savefig(time_plot)
    plt.close()
    
    # 3. EDP by kernel
    plt.figure(figsize=(12, 6))
    edp_sorted_df = df.sort_values('edp')
    
    bars = plt.bar(edp_sorted_df['kernel'], edp_sorted_df['edp'], color='lightsalmon')
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{height:.8f}',
                ha='center', va='bottom', rotation=0)
    
    plt.xlabel("Kernel Implementation")
    plt.ylabel("Energy-Delay Product")
    plt.title("Energy-Delay Product by Kernel Implementation")
    plt.xticks(rotation=45)
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    edp_plot = output_dir / f"kernel_edp_{timestamp}.png"
    plt.savefig(edp_plot)
    plt.close()
    
    # 4. Analysis of optimization techniques
    # Extract all unique techniques
    all_techniques = []
    for techniques in optimization_techniques.values():
        all_techniques.extend(techniques)
    unique_techniques = list(set(all_techniques))
    
    # Create a dataframe to analyze technique effectiveness
    technique_stats = []
    
    for technique in unique_techniques:
        # Find kernels using this technique
        kernels_with_technique = [k for k, techs in optimization_techniques.items() if technique in techs]
        
        # Calculate average metrics for these kernels
        technique_df = df[df['kernel'].isin(kernels_with_technique)]
        
        if not technique_df.empty:
            avg_energy = technique_df['energy_joules'].mean()
            avg_edp = technique_df['edp'].mean()
            avg_power = technique_df['power_watts'].mean()
            avg_time = technique_df['execution_time_sec'].mean()
            
            technique_stats.append({
                'technique': technique,
                'avg_energy': avg_energy,
                'avg_edp': avg_edp,
                'avg_power': avg_power,
                'avg_time': avg_time,
                'kernels_count': len(technique_df)
            })
    
    # Convert to DataFrame and sort by energy efficiency
    technique_df = pd.DataFrame(technique_stats).sort_values('avg_energy')
    
    # Save the technique analysis
    technique_file = output_dir / f"technique_analysis_{timestamp}.csv"
    technique_df.to_csv(technique_file, index=False)
    print(f"Optimization technique analysis saved to {technique_file}")
    
    # Visualization of technique impact on energy
    plt.figure(figsize=(12, 8))
    
    bars = plt.barh(technique_df['technique'], technique_df['avg_energy'], color='skyblue')
    
    # Add value labels on bars
    for bar in bars:
        width = bar.get_width()
        plt.text(width + 0.01, bar.get_y() + bar.get_height()/2.,
                f'{width:.6f}',
                ha='left', va='center')
    
    plt.xlabel('Average Energy Consumption (Joules)')
    plt.ylabel('Optimization Technique')
    plt.title('Energy Impact of Different Optimization Techniques')
    plt.grid(True, axis='x', linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    technique_plot = output_dir / f"optimization_impact_{timestamp}.png"
    plt.savefig(technique_plot)
    plt.close()
    
    # 5. Impact visualization: correlation between techniques and metrics
    plt.figure(figsize=(14, 10))
    plt.subplot(2, 1, 1)
    
    # Sort by energy impact
    energy_impact_df = technique_df.sort_values('avg_energy')
    plt.barh(energy_impact_df['technique'], energy_impact_df['avg_energy'], color='skyblue')
    plt.xlabel('Average Energy (Joules)')
    plt.title('Energy Impact of Optimization Techniques (Lower is Better)')
    plt.grid(True, axis='x', linestyle='--', alpha=0.7)
    
    plt.subplot(2, 1, 2)
    
    # Sort by execution time
    time_impact_df = technique_df.sort_values('avg_time')
    plt.barh(time_impact_df['technique'], time_impact_df['avg_time'], color='lightgreen')
    plt.xlabel('Average Execution Time (seconds)')
    plt.title('Performance Impact of Optimization Techniques (Lower is Better)')
    plt.grid(True, axis='x', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    
    impact_plot = output_dir / f"technique_impact_{timestamp}.png"
    plt.savefig(impact_plot)
    plt.close()
    
    # Print summary of findings
    print("\n===== OPTIMIZATION TECHNIQUE ANALYSIS =====")
    print("Most Energy Efficient Techniques:")
    
    best_energy_techniques = technique_df.head(3)
    for _, row in best_energy_techniques.iterrows():
        print(f"  {row['technique']}: {row['avg_energy']:.6f} Joules")
    
    print("\nMost Performance Efficient Techniques:")
    best_perf_techniques = technique_df.sort_values('avg_time').head(3)
    for _, row in best_perf_techniques.iterrows():
        print(f"  {row['technique']}: {row['avg_time']:.6f} seconds")
    
    print("\nBest Energy-Delay Product (EDP) Techniques:")
    best_edp_techniques = technique_df.sort_values('avg_edp').head(3)
    for _, row in best_edp_techniques.iterrows():
        print(f"  {row['technique']}: {row['avg_edp']:.8f}")
    
    # Rank the kernel implementations
    print("\n===== KERNEL IMPLEMENTATION RANKING =====")
    
    print("Best Energy Efficiency:")
    energy_best = df.sort_values('energy_joules').head(3)
    for _, row in energy_best.iterrows():
        print(f"  {row['kernel']}: {row['energy_joules']:.6f} Joules")
    
    print("\nBest Performance:")
    time_best = df.sort_values('execution_time_sec').head(3)
    for _, row in time_best.iterrows():
        print(f"  {row['kernel']}: {row['execution_time_sec']:.6f} seconds")
    
    print("\nBest Energy-Delay Product (EDP):")
    edp_best = df.sort_values('edp').head(3)
    for _, row in edp_best.iterrows():
        print(f"  {row['kernel']}: {row['edp']:.8f}")
    
    print("\nAnalysis plots saved to:")
    print(f"  - {energy_plot}")
    print(f"  - {time_plot}")
    print(f"  - {edp_plot}")
    print(f"  - {technique_plot}")
    print(f"  - {impact_plot}")
    
    return df, technique_df

if __name__ == "__main__":
    analyze_optimization_techniques()
