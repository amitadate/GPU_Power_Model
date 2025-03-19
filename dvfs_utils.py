import subprocess
import time
import pandas as pd
import numpy as np
import sys
from pathlib import Path

# Add the parent directory to sys.path
sys.path.append(str(Path(__file__).parent.parent))

from gpu_utils import get_gpu_power, get_gpu_utilization

def get_gpu_clocks():
    try:
        # Get SM (Graphics) clock
        result_sm = subprocess.run(
            ['nvidia-smi', '--query-gpu=clocks.current.sm', '--format=csv,noheader,nounits'], 
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True, check=True
        )
        sm_clock = float(result_sm.stdout.strip())
        
        # Get Memory clock
        result_mem = subprocess.run(
            ['nvidia-smi', '--query-gpu=clocks.current.memory', '--format=csv,noheader,nounits'], 
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True, check=True
        )
        mem_clock = float(result_mem.stdout.strip())
        
        return sm_clock, mem_clock
    except (subprocess.CalledProcessError, ValueError) as e:
        print(f"Error getting GPU clock frequencies: {e}")
        return None, None

def get_gpu_pstate():
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=pstate', '--format=csv,noheader,nounits'], 
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True, check=True
        )
        return result.stdout.strip()
    except subprocess.CalledProcessError as e:
        print(f"Error getting GPU P-state: {e}")
        return None

def get_gpu_power_limit():
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=power.limit', '--format=csv,noheader,nounits'], 
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True, check=True
        )
        return float(result.stdout.strip())
    except (subprocess.CalledProcessError, ValueError) as e:
        print(f"Error getting GPU power limit: {e}")
        return None

def collect_dvfs_data(duration=60, interval=1):
    data = []
    start_time = time.time()
    
    print(f"Collecting DVFS data for {duration} seconds...")
    
    while time.time() - start_time < duration:
        timestamp = time.time() - start_time
        util = get_gpu_utilization()
        power = get_gpu_power()
        sm_clock, mem_clock = get_gpu_clocks()
        pstate = get_gpu_pstate()
        
        if None not in [util, power, sm_clock, mem_clock]:
            data.append([timestamp, util, power, sm_clock, mem_clock, pstate])
            
            # Print status every 5 seconds
            if int(timestamp) % 5 == 0 and timestamp > 0 and timestamp < 5.5:
                print(f"Sample data: {util:.1f}% util, {power:.1f}W, {sm_clock:.0f}MHz SM clock")
        
        # Sleep for the specified interval
        time.sleep(interval)
    
    # Create DataFrame
    columns = ['timestamp', 'utilization', 'power', 'sm_clock', 'mem_clock', 'pstate']
    df = pd.DataFrame(data, columns=columns)
    
    print(f"Collected {len(df)} data points.")
    return df

def generate_variable_load(duration=60):
    try:
        import torch
        import math
        import random
        
        if not torch.cuda.is_available():
            print("CUDA is not available. Cannot generate GPU load.")
            return
        
        print(f"Generating variable GPU load for {duration} seconds...")
        device = torch.device('cuda')
        
        # Create tensors of different sizes
        tensors = []
        tensor_sizes = [1000, 2000, 4000, 6000, 8000]
        
        for size in tensor_sizes:
            tensors.append((
                torch.rand(size, size, device=device),
                torch.rand(size, size, device=device)
            ))
        
        # Run workload with varying intensity to trigger different clocks
        start_time = time.time()
        
        while time.time() - start_time < duration:
            # Randomly choose intensity level (which tensors to use)
            level = random.randint(0, len(tensor_sizes) - 1)
            a, b = tensors[level]
            
            # Compute with varying intensity
            if random.random() < 0.7:  # 70% chance of high-intensity operation
                # High-intensity: matrix multiply
                c = torch.matmul(a, b)
                torch.cuda.synchronize()
            else:
                # Lower-intensity means we should do element-wise operations
                c = torch.sin(a) + torch.cos(b)
                torch.cuda.synchronize()
            
            # Vary the sleep time to create different utilization levels
            sleep_time = random.uniform(0.1, 2.0)
            time.sleep(sleep_time)
    
    except ImportError:
        print("PyTorch not found. Cannot generate variable GPU load.")
    except Exception as e:
        print(f"Error in generate_variable_load: {e}")

if __name__ == "__main__":
    # Test the DVFS utilities
    sm_clock, mem_clock = get_gpu_clocks()
    pstate = get_gpu_pstate()
    power_limit = get_gpu_power_limit()
    
    print(f"Current SM Clock: {sm_clock} MHz")
    print(f"Current Memory Clock: {mem_clock} MHz")
    print(f"Current P-State: {pstate}")
    print(f"Power Limit: {power_limit} W")
    
    # Collect a small sample of data
    df = collect_dvfs_data(duration=10)
    print("\nSample data:")
    print(df.head())
