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

# Globals for data collection
terminate_monitoring = False
monitoring_data = []
current_kernel = "None"  

def power_monitoring_thread():
    """function for power monitoring"""
    global monitoring_data, terminate_monitoring, current_kernel
    
    print("Power monitoring thread started")
    
    # Set up output file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path("power_results")
    output_dir.mkdir(exist_ok=True)
    output_file = output_dir / f"histogram_power_data_{timestamp}.csv"
    
    # Open file and write header
    with open(output_file, 'w') as f:
        f.write("timestamp,kernel,power_watts,utilization,sm_clock,mem_clock,temperature,pstate\n")
    
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
            
            # Check for valid readings
            if None not in [power, util, sm_clock, mem_clock, temp]:
                # Record data
                data_point = {
                    "timestamp": current_time,
                    "kernel": current_kernel,
                    "power_watts": power,
                    "utilization": util,
                    "sm_clock": sm_clock,
                    "mem_clock": mem_clock,
                    "temperature": temp,
                    "pstate": pstate
                }
                monitoring_data.append(data_point)
                
                # Write to file immediately to prevent data loss
                with open(output_file, 'a') as f:
                    f.write(f"{current_time},{current_kernel},{power},{util},{sm_clock},{mem_clock},{temp},{pstate}\n")
                
                # Print status every 2 seconds
                if current_time - last_status_time >= 2:
                    last_status_time = current_time
                    print(f"[{current_time:.1f}s] Monitoring {current_kernel}: {power:.1f}W, {util:.1f}%, {sm_clock:.0f}MHz")
            
            # Sleep to control sampling rate (aim for ~10 samples per second)
            time.sleep(0.1)
            
        except Exception as e:
            print(f"Error in monitoring thread: {e}")
            time.sleep(1)  # Sleep longer if there's an error
    
    print(f"Power monitoring completed. Data saved to {output_file}")
    return output_file

def run_individual_kernels():
    global current_kernel
    
    print("\nRunning individual kernel implementations for timing and power analysis")
    
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

def analyze_power_data(kernel_results):
    """Analyze the collected power data and create visualizations."""
    global monitoring_data
    
    if not monitoring_data:
        print("No power data collected")
        return
    
    # Convert to DataFrame
    df = pd.DataFrame(monitoring_data)
    
    # Create a directory for the plots
    plots_dir = Path("power_plots")
    plots_dir.mkdir(exist_ok=True)
    
    # Create a timestamp for the filenames
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Calculate power statistics for each kernel
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
    stats_file = f"power_results/histogram_power_stats_{timestamp}.csv"
    stats_df.to_csv(stats_file, index=False)
    print(f"Power statistics saved to {stats_file}")
    
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
    
    # Create energy comparison bar chart
    plt.figure(figsize=(10, 6))
    plt.bar(stats_df["kernel"], stats_df["energy_joules"])
    plt.xlabel("Kernel Implementation")
    plt.ylabel("Energy Consumption (Joules)")
    plt.title("Energy Consumption by Kernel Implementation")
    plt.xticks(rotation=45)
    plt.grid(True, axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()
    energy_plot = f"power_plots/energy_comparison_{timestamp}.png"
    plt.savefig(energy_plot)
    
    # Create execution time comparison
    plt.figure(figsize=(10, 6))
    plt.bar(stats_df["kernel"], stats_df["execution_time_sec"])
    plt.xlabel("Kernel Implementation")
    plt.ylabel("Execution Time (seconds)")
    plt.title("Execution Time by Kernel Implementation")
    plt.xticks(rotation=45)
    plt.grid(True, axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()
    time_plot = f"power_plots/execution_time_{timestamp}.png"
    plt.savefig(time_plot)
    
    # Create Energy-Delay Product comparison
    plt.figure(figsize=(10, 6))
    plt.bar(stats_df["kernel"], stats_df["edp"])
    plt.xlabel("Kernel Implementation")
    plt.ylabel("Energy-Delay Product (Joule·seconds)")
    plt.title("Energy-Delay Product by Kernel Implementation")
    plt.xticks(rotation=45)
    plt.grid(True, axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()
    edp_plot = f"power_plots/edp_comparison_{timestamp}.png"
    plt.savefig(edp_plot)
    
    # Create clock frequency vs. power scatter plot
    plt.figure(figsize=(10, 6))
    for kernel_name in set(df["kernel"]):
        if kernel_name == "None":
            continue
            
        kernel_df = df[df["kernel"] == kernel_name]
        plt.scatter(kernel_df["sm_clock"], kernel_df["power_watts"], label=kernel_name, alpha=0.5)
    
    plt.xlabel("SM Clock Frequency (MHz)")
    plt.ylabel("Power (Watts)")
    plt.title("Power vs. Clock Frequency")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.7)
    clock_power_plot = f"power_plots/clock_vs_power_{timestamp}.png"
    plt.savefig(clock_power_plot)
    
    # Print summary of results
    print("\nKernel Performance Summary:")
    print(stats_df.to_string(index=False))
    
    print("\nPlots saved to:")
    print(f"  - {power_profile_plot}")
    print(f"  - {energy_plot}")
    print(f"  - {time_plot}")
    print(f"  - {edp_plot}")
    print(f"  - {clock_power_plot}")
    
    return stats_df

def main():
    global terminate_monitoring, monitoring_data, current_kernel
    
    # Create directories for results
    Path("power_results").mkdir(exist_ok=True)
    Path("power_plots").mkdir(exist_ok=True)
    
    # Start power monitoring in a separate thread
    monitoring_thread = threading.Thread(target=power_monitoring_thread)
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
    stats_df = analyze_power_data(kernel_results)
    
    print("\nPower measurement harness completed.")

if __name__ == "__main__":
    main()
