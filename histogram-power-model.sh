#!/bin/bash

# Complete Histogram Power Data Collector
# This script automatically collects power data for all histogram strategies

# Set paths - update these to match your environment
LAB3_DIR="/home/asa5078/368/experiment_lab3/pushed_lab3/labs"
cd $LAB3_DIR

# Create output directory with timestamp
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="histogram_power_data_${TIMESTAMP}"
mkdir -p $OUTPUT_DIR

echo "Histogram Power Data Collection"
echo "==============================="
echo "Output directory: $OUTPUT_DIR"

# Log system information
echo "System Information:" > "${OUTPUT_DIR}/system_info.txt"
nvidia-smi --query-gpu=gpu_name,driver_version,vbios_version --format=csv >> "${OUTPUT_DIR}/system_info.txt"
echo "" >> "${OUTPUT_DIR}/system_info.txt"
lscpu | grep "Model name\|CPU MHz\|CPU(s)\|Thread(s)" >> "${OUTPUT_DIR}/system_info.txt"

# Function to run a single histogram strategy
run_strategy() {
    local strategy=$1
    local strategy_name
    
    case $strategy in
        0) strategy_name="Baseline" ;;
        1) strategy_name="PrivateHist" ;;
        2) strategy_name="SharedMem" ;;
        3) strategy_name="InputTile" ;;
        4) strategy_name="BankConflict" ;;
        5) strategy_name="Coalesced" ;;
        6) strategy_name="LargeTile" ;;
        7) strategy_name="LinearIndexing" ;;
        8) strategy_name="LocalHisto" ;;
        9) strategy_name="LocalBins" ;;
        *) strategy_name="Unknown" ;;
    esac
    
    echo ""
    echo "Running Strategy $strategy ($strategy_name)"
    
    # Prepare output files
    METRICS_FILE="${OUTPUT_DIR}/strategy_${strategy}_metrics.csv"
    OUTPUT_FILE="${OUTPUT_DIR}/strategy_${strategy}_output.txt"
    
    # Start power monitoring
    echo "Starting power monitoring..."
    nvidia-smi --query-gpu=timestamp,utilization.gpu,utilization.memory,power.draw,temperature.gpu,clocks.current.sm,clocks.current.memory,memory.used,memory.total,pstate --format=csv -l 0.05 > "$METRICS_FILE" &
    NVIDIA_SMI_PID=$!
    
    # Let monitoring start
    sleep 1
    
    # Record start time
    START_TIME=$(date +%s.%N)
    echo "Start time: $START_TIME" > "$OUTPUT_FILE"
    
    # Run the specific strategy
    # We'll use environment variable to tell the program which strategy to run
    # This requires modifying the test_harness.cpp to respect this variable
    STRATEGY=$strategy ./bin/linux/release/lab3 >> "$OUTPUT_FILE" 2>&1
    
    # Record end time
    END_TIME=$(date +%s.%N)
    echo "End time: $END_TIME" >> "$OUTPUT_FILE"
    DURATION=$(echo "$END_TIME - $START_TIME" | bc)
    echo "Duration: $DURATION seconds" >> "$OUTPUT_FILE"
    
    # Stop power monitoring
    kill $NVIDIA_SMI_PID
    
    # Extract timing information
    grep "Timing" "$OUTPUT_FILE" > "${OUTPUT_DIR}/strategy_${strategy}_timing.txt"
    
    echo "Strategy $strategy complete"
}

# If STRATEGY env var is set, just run that strategy
if [ ! -z "$STRATEGY" ]; then
    echo "Running only strategy $STRATEGY as specified by environment variable"
    run_strategy $STRATEGY
    exit 0
fi

# Run all strategies
echo "Running all histogram strategies..."

# Skip strategy 1 as it's too slow
for strategy in 0 2 3 4 5 6 7 8 9; do
    run_strategy $strategy
    # Cool down period
    echo "Cooling down for 5 seconds..."
    sleep 5
done

# Create summary file
echo "Creating summary file..."
{
    echo "Strategy,ExecutionTime,AvgGPUUtil,AvgMemUtil,AvgPower,MaxPower,AvgTemp,AvgSMClock,AvgMemClock"
    
    # Process each strategy's data
    for strategy in 0 2 3 4 5 6 7 8 9; do
        METRICS_FILE="${OUTPUT_DIR}/strategy_${strategy}_metrics.csv"
        OUTPUT_FILE="${OUTPUT_DIR}/strategy_${strategy}_output.txt"
        
        # Extract timing info (grep for the opt_2dhisto_strategy line, then get the timing number)
        EXEC_TIME=$(grep "opt_2dhisto_strategy_${strategy}" "$OUTPUT_FILE" | grep "Clock Time" | awk '{print $5}')
        
        # Calculate metrics
        AVG_GPU_UTIL=$(tail -n +2 "$METRICS_FILE" | awk -F, '{sum+=$2} END {print sum/NR}')
        AVG_MEM_UTIL=$(tail -n +2 "$METRICS_FILE" | awk -F, '{sum+=$3} END {print sum/NR}')
        AVG_POWER=$(tail -n +2 "$METRICS_FILE" | awk -F, '{sum+=$4} END {print sum/NR}')
        MAX_POWER=$(tail -n +2 "$METRICS_FILE" | awk -F, '{if($4>max) max=$4} END {print max}')
        AVG_TEMP=$(tail -n +2 "$METRICS_FILE" | awk -F, '{sum+=$5} END {print sum/NR}')
        AVG_SM_CLOCK=$(tail -n +2 "$METRICS_FILE" | awk -F, '{sum+=$6} END {print sum/NR}')
        AVG_MEM_CLOCK=$(tail -n +2 "$METRICS_FILE" | awk -F, '{sum+=$7} END {print sum/NR}')
        
        echo "$strategy,$EXEC_TIME,$AVG_GPU_UTIL,$AVG_MEM_UTIL,$AVG_POWER,$MAX_POWER,$AVG_TEMP,$AVG_SM_CLOCK,$AVG_MEM_CLOCK"
    done
} > "${OUTPUT_DIR}/histogram_summary.csv"

echo "Data collection complete. Results saved to $OUTPUT_DIR"
