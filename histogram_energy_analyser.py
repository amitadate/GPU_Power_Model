#!/usr/bin/env python3

import os
import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime

# Add the current directory to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent))

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
    # Create output directory if it doesn't exist
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True, parents=True)
    
    # Timestamp for filenames
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
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
    
    # Create radar chart comparing all metrics
    from matplotlib.path import Path as MplPath
    from matplotlib.projections.polar import PolarAxes
    from matplotlib.projections import register_projection
    from matplotlib.spines import Spine
    
    def radar_factory(num_vars, frame='circle'):
        # Calculate the angle of each axis
        theta = np.linspace(0, 2*np.pi, num_vars, endpoint=False)

        class RadarAxes(PolarAxes):
            name = 'radar'
            
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.set_theta_zero_location('N')
                
            def fill(self, *args, closed=True, **kwargs):
                """Override fill to handle closed polygons."""
                return super().fill(*(args + (closed,)), **kwargs)
                
            def plot(self, *args, **kwargs):
                """Override plot to handle closed polygons."""
                lines = super().plot(*args, **kwargs)
                for line in lines:
                    self._close_line(line)
                return lines
                
            def _close_line(self, line):
                x, y = line.get_data()
                if x.size > 0 and y.size > 0:
                    x = np.concatenate((x, [x[0]]))
                    y = np.concatenate((y, [y[0]]))
                    line.set_data(x, y)
                    
            def set_varlabels(self, labels):
                self.set_thetagrids(np.degrees(theta), labels)
                
            def _gen_axes_patch(self):
                return plt.Circle((0.5, 0.5), 0.5)
                
            def _gen_axes_spines(self):
                spine_type = 'circle'
                verts = unit_poly_verts(theta)
                return {spine_type: Spine(self, spine_type, MplPath(verts))}
                
        register_projection(RadarAxes)
        fig, ax = plt.subplots(figsize=(9, 9), subplot_kw=dict(projection='radar'))
        
        return fig, ax
    
    def unit_poly_verts(theta):
        x0, y0, r = [0.5] * 3
        verts = [(r*np.cos(t) + x0, r*np.sin(t) + y0) for t in theta]
        return verts
    
    # Prepare data for radar chart
    # Use inverse for metrics where lower is better
    radar_metrics = [
        'execution_time_sec',
        'power_watts',
        'energy_joules',
        'edp',
        'ed2p'
    ]
    
    # Create normalized metrics (0-1 scale, lower is better)
    radar_df = metrics_df.copy()
    for metric in radar_metrics:
        max_val = radar_df[metric].max()
        min_val = radar_df[metric].min()
        if max_val > min_val:
            # Normalize and invert (1 is best, 0 is worst)
            radar_df[f'{metric}_radar'] = 1 - ((radar_df[metric] - min_val) / (max_val - min_val))
    
    # Create radar chart
    radar_data = radar_df[[f'{m}_radar' for m in radar_metrics]].values
    
    fig, ax = radar_factory(len(radar_metrics), frame='polygon')
    
    for i, row in enumerate(radar_df.itertuples()):
        ax.plot(radar_data[i], 'o-', linewidth=2, label=row.kernel)
        ax.fill(radar_data[i], alpha=0.1)
    
    ax.set_varlabels([
        'Execution\nTime',
        'Power\nConsumption', 
        'Energy\nConsumption',
        'Energy-Delay\nProduct',
        'Energy-Delay²\nProduct'
    ])
    
    plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
    plt.title('Kernel Efficiency Comparison (further out is better)')
    
    radar_plot = output_path / f"efficiency_radar_{timestamp}.png"
    plt.savefig(radar_plot, bbox_inches='tight')
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
    print(f"  - {radar_plot.name}")
    print(f"  - {pareto_plot.name}")
    
    return {
        'energy_plot': energy_plot,
        'metrics_plot': metrics_plot,
        'radar_plot': radar_plot,
        'pareto_plot': pareto_plot
    }

def get_execution_times_from_csv(csv_path):
    try:
        df = pd.read_csv(csv_path)
        
        # Check if the required columns exist
        if 'kernel' in df.columns and 'execution_time_sec' in df.columns:
            # Create a dictionary mapping kernel names to execution times
            return dict(zip(df['kernel'], df['execution_time_sec']))
        else:
            print(f"CSV file {csv_path} does not contain required columns")
            return None
    except Exception as e:
        print(f"Error reading execution times from {csv_path}: {e}")
        return None

def main():
    parser = argparse.ArgumentParser(description="Histogram Energy Analyzer")
    parser.add_argument("--power-data", type=str, required=True,
                      help="Path to the power data CSV file")
    parser.add_argument("--results", type=str,
                      help="Path to the results CSV file with execution times")
    parser.add_argument("--output-dir", type=str, default="energy_analysis",
                      help="Directory to save analysis results")
    
    args = parser.parse_args()
    
    # Check if power data file exists
    power_data_path = Path(args.power_data)
    if not power_data_path.exists():
        print(f"Error: Power data file {power_data_path} does not exist")
        return
    
    # Load the power data
    try:
        power_data = pd.read_csv(power_data_path)
        print(f"Loaded power data from {power_data_path}")
        print(f"Data contains {len(power_data)} samples across {power_data['kernel'].nunique()} kernels")
    except Exception as e:
        print(f"Error loading power data: {e}")
        return
    
    # Get execution times
    if args.results:
        results_path = Path(args.results)
        if not results_path.exists():
            print(f"Error: Results file {results_path} does not exist")
            return
        
        execution_times = get_execution_times_from_csv(results_path)
        if not execution_times:
            print("Could not get execution times from results file")
            return
    else:
        # infer execution times if cabt fetch
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
            return
        
        execution_times = kernel_times
    
    # Calculate energy metrics
    metrics_df = calculate_energy_metrics(power_data, execution_times)
    
    # Save the metrics
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    metrics_file = output_dir / f"energy_metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
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
