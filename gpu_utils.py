import subprocess
import os
import time
import pynvml
import pandas as pd
import numpy as np
from pathlib import Path

def ensure_dir_structure():
    """Create the necessary directory structure if it doesn't exist."""
    base_dir = Path(__file__).parent.parent
    
    # Create directories
    directories = [
        base_dir / "data" / "raw",
        base_dir / "data" / "processed",
        base_dir / "models",
        base_dir / "plots",
        base_dir / "notebooks"
    ]
    
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)
    
    return base_dir

# Basic GPU information
def get_gpu_power():
    """Get current GPU power draw in watts using nvidia-smi."""
    try:
        #result = subprocess.run(['nvidia-smi', '--query-gpu=power.draw', '--format=csv,noheader,nounits'],capture_output=True, text=True, check=True)
        result = subprocess.run(['nvidia-smi', '--query-gpu=power.draw', '--format=csv,noheader,nounits'], 
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE, 
                                universal_newlines=True, check=True)
        return float(result.stdout.strip())
    except subprocess.CalledProcessError as e:
        print(f"Error getting GPU power: {e}")
        return None
    except ValueError as e:
        print(f"Could not convert power value to float: {e}")
        return None

def get_gpu_utilization():
    """Get current GPU utilization percentage using nvidia-smi."""
    try:
        #result = subprocess.run(['nvidia-smi', '--query-gpu=utilization.gpu', '--format=csv,noheader,nounits'], capture_output=True, text=True, check=True)
        result = subprocess.run(['nvidia-smi', '--query-gpu=utilization.gpu', '--format=csv,noheader,nounits'], 
                                        stdout=subprocess.PIPE, stderr=subprocess.PIPE, 
                                        universal_newlines=True, check=True)
        return float(result.stdout.strip())
    except subprocess.CalledProcessError as e:
        print(f"Error getting GPU utilization: {e}")
        return None
    except ValueError as e:
        print(f"Could not convert utilization value to float: {e}")
        return None

def get_gpu_memory_used():
    """Get current GPU memory usage in MiB using nvidia-smi."""
    try:
        #result = subprocess.run(['nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader,nounits'], capture_output=True, text=True, check=True)
        result = subprocess.run(['nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader,nounits'], 
                                        stdout=subprocess.PIPE, stderr=subprocess.PIPE, 
                                        universal_newlines=True, check=True)
        return float(result.stdout.strip())
    except subprocess.CalledProcessError as e:
        print(f"Error getting GPU memory usage: {e}")
        return None
    except ValueError as e:
        print(f"Could not convert memory value to float: {e}")
        return None

def get_gpu_temperature():
    """Get current GPU temperature in Celsius using nvidia-smi."""
    try:
        #result = subprocess.run(['nvidia-smi', '--query-gpu=temperature.gpu', '--format=csv,noheader,nounits'], capture_output=True, text=True, check=True)
        result = subprocess.run(['nvidia-smi', '--query-gpu=temperature.gpu', '--format=csv,noheader,nounits'], 
                                        stdout=subprocess.PIPE, stderr=subprocess.PIPE, 
                                        universal_newlines=True, check=True)
        return float(result.stdout.strip())
    except subprocess.CalledProcessError as e:
        print(f"Error getting GPU temperature: {e}")
        return None
    except ValueError as e:
        print(f"Could not convert temperature value to float: {e}")
        return None

# NVML functions for more detailed information
def init_nvml():
    """Initialize NVML."""
    try:
        pynvml.nvmlInit()
        return True
    except pynvml.NVMLError as e:
        print(f"Error initializing NVML: {e}")
        return False

def shutdown_nvml():
    """Shutdown NVML."""
    try:
        pynvml.nvmlShutdown()
        return True
    except pynvml.NVMLError as e:
        print(f"Error shutting down NVML: {e}")
        return False

def get_device_handle(device_id=0):
    """Get the handle for the GPU device."""
    try:
        return pynvml.nvmlDeviceGetHandleByIndex(device_id)
    except pynvml.NVMLError as e:
        print(f"Error getting device handle: {e}")
        return None

def get_sm_utilization(handle):
    """Get SM utilization using NVML."""
    try:
        util = pynvml.nvmlDeviceGetUtilizationRates(handle)
        return util.gpu  # Overall GPU utilization
    except pynvml.NVMLError as e:
        print(f"Error getting SM utilization: {e}")
        return None

def get_device_count():
    """Get the number of GPU devices on the system."""
    try:
        return pynvml.nvmlDeviceGetCount()
    except pynvml.NVMLError as e:
        print(f"Error getting device count: {e}")
        return None

def get_device_name(handle):
    """Get the name of the GPU device."""
    try:
        return pynvml.nvmlDeviceGetName(handle)
    except pynvml.NVMLError as e:
        print(f"Error getting device name: {e}")
        return None

def get_rtx2080_super_specs():
    """Return specifications for RTX 2080 Super."""
    return {
        "name": "NVIDIA GeForce RTX 2080 SUPER",
        "num_sms": 48,
        "cuda_cores": 3072,  # 48 SMs × 64 CUDA Cores per SM
        "memory": 8192,  # 8 GB GDDR6
        "tdp": 250,  # 250W TDP
    }

if __name__ == "__main__":
    # Test the utilities
    base_dir = ensure_dir_structure()
    print(f"Project directory structure created at: {base_dir}")
    
    print("\nTesting GPU utilities:")
    print(f"Current GPU Power: {get_gpu_power()} W")
    print(f"Current GPU Utilization: {get_gpu_utilization()} %")
    print(f"Current GPU Memory Used: {get_gpu_memory_used()} MiB")
    print(f"Current GPU Temperature: {get_gpu_temperature()} °C")
    
    if init_nvml():
        handle = get_device_handle()
        if handle:
            print(f"\nGPU Device: {get_device_name(handle)}")
            print(f"SM Utilization: {get_sm_utilization(handle)} %")
        shutdown_nvml()
    
    rtx_specs = get_rtx2080_super_specs()
    print(f"\nRTX 2080 Super specs:")
    for key, value in rtx_specs.items():
        print(f"  {key}: {value}")
