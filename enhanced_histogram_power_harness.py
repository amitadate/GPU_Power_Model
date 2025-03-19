#!/usr/bin/env python3

import os
import sys
import time
import threading
import subprocess
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime

# Import from your existing GPU power modeling code
from gpu_utils import get_gpu_power, get_gpu_utilization, get_gpu_temperature
from dvfs_utils import get_gpu_clocks, get_gpu_pstate
from memory_utils import get_memory_metrics_nvidia_smi, get_memory_bandwidth_usage

# Globals for data collection
terminate_monitoring = False
monitoring_data = []
current_kernel = "None" 

def enhanced_power_monitoring_thread():
    """Thread function that continuously monitors GPU power, frequency, and memory metrics."""
    global monitoring_data, terminate_monitoring, current_kernel
    
    print("Enhanced power monitoring thread started")
    
    # Set up output file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path("power_results")
    output_dir.mkdir(exist_ok=True)
    output_file = output_dir / f"enhanced_histogram_power_data_{timestamp}.csv"
    
    with open(output_file, 'w') as f:
        f.write("timestamp,kernel,power_watts,utilization,sm_clock,mem_clock,temperature,pstate,memory_utilization,memory_used_MiB,memory_total_MiB,memory_free_MiB,memory_bandwidth_pct\n")
    
    # Start monitoring loop
    start_time = time.time()
    last_status_time = 0
    
    while not terminate_monitoring:
        try:
            # Get current measurements
            current_time = time.time() - start_time
            power = get_gpu_power()
            util = get_gpu_utilization()
            sm_clock, mem_clock = get_gpu_clocks()
            temp = get_gpu_temperature()
            pstate = get_gpu_pstate()
            
            # Get memory metrics
            memory_metrics = get_memory_metrics_nvidia_smi()
            memory_bandwidth = get_memory_bandwidth_usage()
            
            # Check for valid readings
            if None not in [power, util, sm_clock, mem_clock, temp] and memory_metrics and memory_bandwidth is not None:
                # Record data
                data_point = {
                    "timestamp": current_time,
                    "kernel": current_kernel,
                    "power_watts": power,
                    "utilization": util,
                    "sm_clock": sm_clock,
                    "mem_clock": mem_clock,
                    "temperature": temp,
                    "pstate": pstate,
                    "memory_utilization": memory_metrics["memory_utilization"],
                    "memory_used_MiB": memory_metrics["memory_used_MiB"],
                    "memory_total_MiB": memory_metrics["memory_total_MiB"],
                    "memory_free_MiB": memory_metrics["memory_free_MiB"],
                    "memory_bandwidth_pct": memory_bandwidth
                }
                monitoring_data.append(data_point)
                
                # Write to file immediately
                with open(output_file, 'a') as f:
                    f.write(f"{current_time},{current_kernel},{power},{util},{sm_clock},{mem_clock},{temp},{pstate},{memory_metrics['memory_utilization']},{memory_metrics['memory_used_MiB']},{memory_metrics['memory_total_MiB']},{memory_metrics['memory_free_MiB']},{memory_bandwidth}\n")
                
                # Print status every 5 seconds
                if int(current_time) % 5 == 0 and current_time > 0 and current_time < 5.5:
                    print(f"[{current_time:.1f}s] Monitoring {current_kernel}: {power:.1f}W, {util:.1f}%, {sm_clock:.0f}MHz, Mem: {memory_metrics['memory_utilization']:.1f}%")
            
            # (aim for ~10 samples per second)
            time.sleep(0.1)
            
        except Exception as e:
            print(f"Error in monitoring thread: {e}")
            time.sleep(1) 
    
    print(f"Enhanced power monitoring completed. Data saved to {output_file}")
    return output_file

def run_individual_kernels():
    global current_kernel
    
    print("\nRunning individual kernel implementations for timing, power and memory analysis")
    
    # Dictionary to store results for each kernel
    kernel_results = {}
    
    # Path to executable
    executable = "../.././bin/linux/release/lab3"
    
    # Kernel implementations to test
    kernel_implementations = [
        "Baseline", 
        "Strategy1", 
        "Strategy2", 
        "Strategy3", 
        "Strategy4",
        "Strategy5", 
        "Strategy6", 
        "Strategy7", 
        "Strategy8", 
        "Strategy9"
    ]
    
    # Run each kernel implementation separately
    for k_id, kernel_name in enumerate(kernel_implementations):
        print(f"\n----------- Testing {kernel_name} ({k_id+1}/{len(kernel_implementations)}) -----------")
        
        # Set the current kernel for power monitoring
        current_kernel = kernel_name
        
        # Wait a moment for monitoring to register the change
        time.sleep(1)
        
        # Run the specific kernel implementation (with kernel ID)
        cmd = [executable, "--kernel", str(k_id)]
        print(f"Executing: {' '.join(cmd)}")
        
        # Time the execution
        start_time = time.time()
        
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
        stdout, stderr = process.communicate()
        
        end_time = time.time()
        execution_time = end_time - start_time
        
        # Capture timing information from the output, if available
        precise_time = None
        for line in stdout.splitlines():
            if "Clock Time" in line:
                try:
                    precise_time = float(line.split('=')[1].strip())
                    break
                except:
                    pass
        
        # Use the more precise timing if available
        if precise_time is not None:
            execution_time = precise_time
        
        # Store results
        kernel_results[kernel_name] = {
            "execution_time": execution_time,
            "return_code": process.returncode,
            "stdout": stdout,
            "stderr": stderr
        }
        
        print(f"{kernel_name} executed in {execution_time:.6f} seconds")
        
        # Wait a moment for power to stabilize before next kernel
        print(f"Waiting for power to stabilize...")
        time.sleep(2)
    
    # Convert results to the expected format
    formatted_results = []
    for kernel_name, result in kernel_results.items():
        formatted_results.append({
            "kernel": kernel_name,
            "execution_time": result["execution_time"],
            "return_code": result["return_code"]
        })
    
    return formatted_results

def analyze_power_and_memory_data(kernel_results):
    global monitoring_data
    
    if not monitoring_data:
        print("No data collected")
        return
    
    # Convert to DataFrame
    df = pd.DataFrame(monitoring_data)
    
    # Create a directory for the plots
    plots_dir = Path("power_plots")
    plots_dir.mkdir(exist_ok=True)
    
    # Create a timestamp for the filenames
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Calculate power and memory statistics for each kernel
    kernel_stats = []
    
    for kernel_name in set(df["kernel"]):
        if kernel_name == "None":
            continue
            
        # Filter data for this kernel
        kernel_df = df[df["kernel"] == kernel_name]
        
        # Calculate statistics
        avg_power = kernel_df["power_watts"].mean()
        max_power = kernel_df["power_watts"].max()
        avg_clock = kernel_df["sm_clock"].mean()
        avg_mem_util = kernel_df["memory_utilization"].mean()
        avg_mem_bandwidth = kernel_df["memory_bandwidth_pct"].mean()
        
        # Get execution time
        execution_time = next((result["execution_time"] for result in kernel_results if result["kernel"] == kernel_name), None)
        
        # Calculate energy consumption (power * time)
        energy = avg_power * execution_time if execution_time else None
        
        # Calculate Energy-Delay Product (EDP)
        edp = energy * execution_time if energy and execution_time else None
        
        kernel_stats.append({
            "kernel": kernel_name,
            "avg_power_watts": avg_power,
            "max_power_watts": max_power,
            "avg_clock_mhz": avg_clock,
            "avg_memory_util_pct": avg_mem_util,
            "avg_memory_bandwidth_pct": avg_mem_bandwidth,
            "execution_time_sec": execution_time,
            "energy_joules": energy,
            "edp": edp
        })
    
    # Create a DataFrame with the statistics
    stats_df = pd.DataFrame(kernel_stats)
    
    # Add normalized values
    if not stats_df.empty and stats_df["energy_joules"].notna().any():
        min_energy = stats_df["energy_joules"].min()
        stats_df["energy_normalized"] = stats_df["energy_joules"] / min_energy
        
        min_edp = stats_df["edp"].min()
        stats_df["edp_normalized"] = stats_df["edp"] / min_edp
    
    # Save the statistics
    stats_file = f"power_results/enhanced_histogram_power_stats_{timestamp}.csv"
    stats_df.to_csv(stats_file, index=False)
    print(f"Power and memory statistics saved to {stats_file}")
    
    # Create power profile plot
    plt.figure(figsize=(12, 8))
    for kernel_name in set(df["kernel"]):
        if kernel_name == "None":
            continue
            
        kernel_df = df[df["kernel"] == kernel_name]
        if not kernel_df.empty:
            # Adjust timestamps to be relative to the start of each kernel
            kernel_df = kernel_df.copy()
            min_time = kernel_df["timestamp"].min()
            kernel_df["relative_time"] = kernel_df["timestamp"] - min_time
            
            plt.plot(kernel_df["relative_time"], kernel_df["power_watts"], label=kernel_name)
    
    plt.xlabel("Time (seconds)")
    plt.ylabel("Power (Watts)")
    plt.title("Power Consumption Over Time")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.7)
    power_profile_plot = f"power_plots/power_profile_{timestamp}.png"
    plt.savefig(power_profile_plot)
    plt.close()
    
    # Create memory utilization plot
    plt.figure(figsize=(12, 8))
    for kernel_name in set(df["kernel"]):
        if kernel_name == "None":
            continue
            
        kernel_df = df[df["kernel"] == kernel_name]
        if not kernel_df.empty:
            # Adjust timestamps to be relative to the start of each kernel
            kernel_df = kernel_df.copy()
            min_time = kernel_df["timestamp"].min()
            kernel_df["relative_time"] = kernel_df["timestamp"] - min_time
            
            plt.plot(kernel_df["relative_time"], kernel_df["memory_utilization"], label=kernel_name)
    
    plt.xlabel("Time (seconds)")
    plt.ylabel("Memory Utilization (%)")
    plt.title("Memory Utilization Over Time")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.7)
    memory_plot = f"power_plots/memory_utilization_{timestamp}.png"
    plt.savefig(memory_plot)
    plt.close()
    
    # Create memory bandwidth plot
    plt.figure(figsize=(12, 8))
    for kernel_name in set(df["kernel"]):
        if kernel_name == "None":
            continue
            
        kernel_df = df[df["kernel"] == kernel_name]
        if not kernel_df.empty:
            # Adjust timestamps to be relative to the start of each kernel
            kernel_df = kernel_df.copy()
            min_time = kernel_df["timestamp"].min()
            kernel_df["relative_time"] = kernel_df["timestamp"] - min_time
            
            plt.plot(kernel_df["relative_time"], kernel_df["memory_bandwidth_pct"], label=kernel_name)
    
    plt.xlabel("Time (seconds)")
    plt.ylabel("Memory Bandwidth Utilization (%)")
    plt.title("Memory Bandwidth Over Time")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.7)
    bandwidth_plot = f"power_plots/memory_bandwidth_{timestamp}.png"
    plt.savefig(bandwidth_plot)
    plt.close()
    
    # Create correlation plot: Power vs Memory Bandwidth
    plt.figure(figsize=(10, 6))
    for kernel_name in set(df["kernel"]):
        if kernel_name == "None":
            continue
            
        kernel_df = df[df["kernel"] == kernel_name]
        if not kernel_df.empty:
            plt.scatter(kernel_df["memory_bandwidth_pct"], kernel_df["power_watts"], label=kernel_name, alpha=0.5)
    
    plt.xlabel("Memory Bandwidth Utilization (%)")
    plt.ylabel("Power (Watts)")
    plt.title("Power vs Memory Bandwidth")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.7)
    corr_plot = f"power_plots/power_vs_memory_{timestamp}.png"
    plt.savefig(corr_plot)
    plt.close()
    
    # Print summary of results
    print("\nKernel Performance Summary:")
    print(stats_df[["kernel", "avg_power_watts", "avg_memory_util_pct", "avg_memory_bandwidth_pct", "execution_time_sec", "energy_joules"]].to_string(index=False))
    
    print("\nPlots saved to:")
    print(f"  - {power_profile_plot}")
    print(f"  - {memory_plot}")
    print(f"  - {bandwidth_plot}")
    print(f"  - {corr_plot}")
    
    return stats_df

def main():
    global terminate_monitoring, monitoring_data, current_kernel
    
    # Create directories for results
    Path("power_results").mkdir(exist_ok=True)
    Path("power_plots").mkdir(exist_ok=True)
    
    # Start enhanced power monitoring in a separate thread
    monitoring_thread = threading.Thread(target=enhanced_power_monitoring_thread)
    monitoring_thread.daemon = True
    monitoring_thread.start()
    
    # Wait for the monitoring thread to start
    time.sleep(2)
    
    try:
        # Run individual kernels and collect timing
        kernel_results = run_individual_kernels()
        
        # Set current_kernel back to None
        current_kernel = "None"
        
        # Wait for monitoring to stabilize
        time.sleep(1)
        
    finally:
        # Stop power monitoring
        terminate_monitoring = True
        monitoring_thread.join()
    
    # Analyze the collected data
    stats_df = analyze_power_and_memory_data(kernel_results)
    
    print("\nEnhanced power measurement harness completed.")

if __name__ == "__main__":
    main()
