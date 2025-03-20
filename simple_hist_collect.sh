#!/bin/bash

# Simple Histogram Power Collector
# This script collects power data while running the histogram workload

# Change to your directory where the lab3 executable is
cd /home/asa5078/368/experiment_lab3/pushed_lab3/labs

# Create a directory for the power data
mkdir -p histogram_power_data

# Start collecting power data
echo "Starting power data collection..."
nvidia-smi --query-gpu=timestamp,utilization.gpu,utilization.memory,power.draw,temperature.gpu,clocks.current.sm,clocks.current.memory --format=csv -l 0.05 > histogram_power_data/histogram_power.csv &
NVIDIA_SMI_PID=$!

# Wait a moment for nvidia-smi to start
sleep 1

# Run the histogram workload
echo "Running histogram workload..."
./bin/linux/release/lab3 | tee histogram_power_data/histogram_output.txt

# Stop collecting power data
kill $NVIDIA_SMI_PID
echo "Power data collection complete. Data saved to histogram_power_data/histogram_power.csv"
