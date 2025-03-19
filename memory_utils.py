#!/usr/bin/env python3

import subprocess
import json
import pandas as pd
import numpy as np
import sys
from pathlib import Path

# Add the parent directory to sys.path if needed
sys.path.append(str(Path(__file__).parent.parent))

def get_memory_metrics_nsight(kernel_name):

    metrics = [
        "dram__bytes_read.sum",
        "dram__bytes_write.sum",
        "l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum",
        "l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum",
        "lts__t_sectors_op_read.sum",
        "lts__t_sectors_op_write.sum",
        "gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed",
        "l1tex__data_pipe_lsu_wavefronts_mem_shared.avg",
        "sm__warps_active.avg.pct_of_peak_sustained_active"
    ]
    
    metrics_str = ",".join(metrics)
    
    try:
        cmd = [
            "nv-nsight-cu-cli", 
            "--metrics", metrics_str,
            "--csv", 
            "--kernel-name", kernel_name,
            "your_application" #binary path
        ]
        
        # Use Popen instead of run with capture_output
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
        stdout, stderr = process.communicate()
        
        if process.returncode != 0:
            print(f"Error running Nsight Compute: {stderr}")
            return {}
        
        # Parse the CSV output
        lines = stdout.strip().split('\n')
        header = lines[0].split(',')
        values = lines[1].split(',')
        
        return dict(zip(header, values))
    
    except Exception as e:
        print(f"Error running Nsight Compute: {e}")
        return {}

def get_memory_metrics_nvidia_smi():
    try:
        cmd = [
            "nvidia-smi", 
            "--query-gpu=utilization.memory,memory.used,memory.total,memory.free",
            "--format=csv,noheader,nounits"
        ]
        
        # Use Popen instead of run with capture_output
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
        stdout, stderr = process.communicate()
        
        if process.returncode != 0:
            print(f"Error executing nvidia-smi: {stderr}")
            return {}
        
        values = stdout.strip().split(',')
        
        if len(values) >= 4:
            return {
                "memory_utilization": float(values[0]),
                "memory_used_MiB": float(values[1]),
                "memory_total_MiB": float(values[2]),
                "memory_free_MiB": float(values[3])
            }
        return {}
    
    except Exception as e:
        print(f"Error getting GPU memory metrics: {e}")
        return {}

def get_memory_bandwidth_usage():
    try:
        cmd = ["nvidia-smi", "--query-gpu=utilization.memory", "--format=csv,noheader,nounits"]
        
        # Use Popen instead of run with capture_output
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
        stdout, stderr = process.communicate()
        
        if process.returncode != 0:
            print(f"Error executing nvidia-smi: {stderr}")
            return None
            
        return float(stdout.strip())
    except (Exception, ValueError) as e:
        print(f"Error getting memory bandwidth usage: {e}")
        return None

def collect_memory_metrics(duration=5, interval=0.5):
    import time
    
    data = []
    start_time = time.time()
    
    print(f"Collecting memory metrics for {duration} seconds...")
    
    while time.time() - start_time < duration:
        timestamp = time.time() - start_time
        
        # Get general memory metrics
        memory_metrics = get_memory_metrics_nvidia_smi()
        bandwidth = get_memory_bandwidth_usage()
        
        if memory_metrics and bandwidth is not None:
            row = {
                "timestamp": timestamp,
                "memory_bandwidth_pct": bandwidth,
            }
            row.update(memory_metrics)
            data.append(row)
        
        time.sleep(interval)
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    print(f"Collected {len(df)} memory metric samples.")
    return df

if __name__ == "__main__":
    # Test the utilities
    print("Testing memory metrics collection:")
    df = collect_memory_metrics(duration=3)
    print(df.head())
    
    # print("\nKernel memory metrics:")
    # metrics = get_memory_metrics_nsight("YourKernelName")
    # for key, value in metrics.items():
    #     print(f"  {key}: {value}")
