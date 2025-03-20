#!/bin/bash

# Basic GPU Power Data Collector
# A minimal script that just collects power data while you run your workload manually

# Usage: ./basic-power-collector.sh output_filename.csv
# Press Ctrl+C to stop data collection when your workload is complete

OUTPUT_FILE=$1

if [ -z "$OUTPUT_FILE" ]; then
    OUTPUT_FILE="gpu_power_data.csv"
fi

echo "Starting power data collection. Data will be saved to $OUTPUT_FILE"
echo "Run your workload in another terminal window."
echo "Press Ctrl+C to stop data collection when finished."

# Start collecting data with nvidia-smi at 100ms intervals
nvidia-smi --query-gpu=timestamp,utilization.gpu,utilization.memory,power.draw,temperature.gpu,clocks.current.sm,clocks.current.memory --format=csv -l 0.1 > "$OUTPUT_FILE"

# This will run until user presses Ctrl+C
