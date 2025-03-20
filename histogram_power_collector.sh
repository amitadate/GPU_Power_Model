#!/bin/bash

# Histogram Power Data Collector
# This script collects power data while running the modified histogram test_harness

# Path to your lab3 directory - update this to match your environment
LAB3_DIR="/home/asa5078/368/experiment_lab3/pushed_lab3/labs"
cd $LAB3_DIR

# Create a directory for the power data
POWER_DIR="histogram_power_data"
mkdir -p $POWER_DIR

# Start power data collection
echo "Starting power data collection..."
nvidia-smi --query-gpu=timestamp,utilization.gpu,utilization.memory,power.draw,temperature.gpu,clocks.current.sm,clocks.current.memory --format=csv -l 0.05 > "$POWER_DIR/histogram_all_strategies.csv" &
NVIDIA_SMI_PID=$!

# Wait a moment for the power monitoring to start
sleep 1

# Run the test_harness which will now run all strategies
echo "Running histogram with all strategies..."
./bin/linux/release/lab3 | tee "$POWER_DIR/histogram_output.txt"

# Stop power monitoring
kill $NVIDIA_SMI_PID
echo "Power data collection complete. Data saved to $POWER_DIR/histogram_all_strategies.csv"
