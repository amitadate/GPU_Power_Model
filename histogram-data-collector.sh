#!/bin/bash

# Histogram Data Collection Script adjusted for your environment
# This script runs the histogram workload with various strategies while collecting GPU metrics

# Navigate to the directory containing test_harness and opt_2dhisto
# Update this path to match your histogram directory
WORKDIR="/home/asa5078/368/experiment_lab3/pushed_lab3/labs/src/lab3"
cd $WORKDIR

# Create a directory for the collected data
DATA_DIR="histogram_power_data"
mkdir -p $DATA_DIR

# Function to run an experiment with a specific kernel strategy
run_kernel_experiment() {
    local strategy=$1
    local output_file="$DATA_DIR/histogram_power_strategy_${strategy}.csv"
    local metrics_file="$DATA_DIR/histogram_metrics_strategy_${strategy}.csv"
    
    echo "Running experiment with strategy $strategy..."
    
    # Start nvidia-smi in the background with 50ms sampling interval
    nvidia-smi --query-gpu=timestamp,utilization.gpu,utilization.memory,power.draw,temperature.gpu,clocks.current.sm,clocks.current.memory --format=csv -l 0.05 > "$metrics_file" &
    NVIDIA_SMI_PID=$!
    
    # Sleep briefly to ensure nvidia-smi starts collecting before the workload
    sleep 1
    
    # Record the start time
    echo "Start time: $(date +%s.%N)" > "$output_file"
    
    # Modify opt_2dhisto.cu to use a specific kernel strategy
    # First, make a backup of the file
    cp opt_2dhisto.cu opt_2dhisto.cu.backup
    
    # Comment out all kernel calls
    sed -i 's/^[ \t]*\(.*ParallelStrategy[A-Za-z]*<<<\)/\/\/\1/g' opt_2dhisto.cu
    sed -i 's/^[ \t]*\(.*Baseline<<<\)/\/\/\1/g' opt_2dhisto.cu
    
    # Uncomment the specific kernel strategy
    case $strategy in
        baseline)
            # Uncomment the baseline kernel
            sed -i '/\/\/\/\/\/\/\/\/\/ --- baseline kernel ---/,/\/\/\/\/\/\/ --- baseline kernel ---/s/\/\/\(.*Baseline<<<.*\)/\1/g' opt_2dhisto.cu
            ;;
        1)
            # Uncomment ParallelStrategyOne kernel
            sed -i '/\/\/\/\/\/\/\/\/\/\/ -- ParallelStrategyOne kernel ---/,/\/\/\/\/\/\/\/\/\/ -- ParallelStrategyOne kernel ---/s/\/\/\(.*ParallelStrategyOne<<<.*\)/\1/g' opt_2dhisto.cu
            ;;
        2)
            # Uncomment ParallelStrategyTwo kernel
            sed -i '/\/\/\/\/\/\/ --- ParallelStrategyTwo kernel ---/,/\/\/\/\/\/\/ --- ParallelStrategyTwo kernel ---/s/\/\/\(.*ParallelStrategyTwo<<<.*\)/\1/g' opt_2dhisto.cu
            ;;
        3)
            # Uncomment ParallelStrategyThree kernel
            sed -i '/\/\/\/\/\/\/ --- ParallelStrategyThree kernel ---/,/\/\/\/\/\/\/ --- ParallelStrategyThree kernel ---/s/\/\/\(.*ParallelStrategyThree<<<.*\)/\1/g' opt_2dhisto.cu
            ;;
        4)
            # Uncomment ParallelStrategyFour kernel
            sed -i '/\/\/\/\/\/\/ --- ParallelStrategyFour  kernel ---/,/\/\/\/\/\/\/ --- ParallelStrategyFour kernel ---/s/\/\/\(.*ParallelStrategyFour<<<.*\)/\1/g' opt_2dhisto.cu
            ;;
        5)
            # Uncomment ParallelStrategyFive kernel
            sed -i '/\/\/\/\/\/\/ --- ParallelStrategyFive  kernel ---/,/\/\/\/\/\/\/ --- ParallelStrategyFive kernel ---/s/\/\/\(.*ParallelStrategyFive<<<.*\)/\1/g' opt_2dhisto.cu
            ;;
        6)
            # Uncomment ParallelStrategySix kernel
            sed -i '/\/\/\/\/\/\/ --- ParallelStrategySix kernel ---/,\/\/\/\/\/\/ -- ParallelStrategySix  kernel ---/s/\/\/\(.*ParallelStrategySix<<<.*\)/\1/g' opt_2dhisto.cu
            ;;
        7)
            # Uncomment ParallelStrategySeven kernel
            sed -i '/\/\/\/\/\/\/ --- ParallelStrategySeven kernel ---/,/\/\/\/\/\/\/ --- ParallelStrategySeven kernel ---/s/\/\/\(.*ParallelStrategySeven<<<.*\)/\1/g' opt_2dhisto.cu
            ;;
        8)
            # Uncomment ParallelStrategyEight kernel
            sed -i '/\/\/\/\/\/\/ --- ParallelStrategyEight kernel ---/,/\/\/\/\/\/\/ --- ParallelStrategyEight kernel ---/s/\/\/\(.*ParallelStrategyEight<<<.*\)/\1/g' opt_2dhisto.cu
            ;;
        9)
            # Uncomment ParallelStrategyNine kernel
            sed -i '/\/\/\/\/\/\/ --- ParallelStrategyNine kernel ---/,/\/\/\/\/\/\/ --- ParallelStrategyNine kernel ---/s/\/\/\(.*ParallelStrategyNine<<<.*\)/\1/g' opt_2dhisto.cu
            ;;
    esac
    
    # Rebuild and run the histogram executable
    make clean >> "$output_file" 2>&1
    make >> "$output_file" 2>&1
    ./test_harness >> "$output_file" 2>&1
    
    # Record the end time
    echo "End time: $(date +%s.%N)" >> "$output_file"
    
    # Restore the original file
    mv opt_2dhisto.cu.backup opt_2dhisto.cu
    
    # Kill the nvidia-smi process
    kill $NVIDIA_SMI_PID
    
    # Extract important metrics from the output
    grep -A 2 "opt_2dhisto" "$output_file" >> "$DATA_DIR/histogram_results.txt"
    grep "Test" "$output_file" >> "$DATA_DIR/histogram_results.txt"
    echo "------------------------------------------------------" >> "$DATA_DIR/histogram_results.txt"
    
    # Wait for a cooldown period between experiments
    echo "Cooldown period..."
    sleep 5
}

# Function to run experiments with different seeds to generate different input patterns
run_seed_experiment() {
    local seed=$1
    local strategy=$2  # Use the best strategy for all seed tests
    local output_file="$DATA_DIR/histogram_power_seed_${seed}_strategy_${strategy}.csv"
    local metrics_file="$DATA_DIR/histogram_metrics_seed_${seed}_strategy_${strategy}.csv"
    
    echo "Running experiment with seed $seed using strategy $strategy..."
    
    # Start nvidia-smi in the background
    nvidia-smi --query-gpu=timestamp,utilization.gpu,utilization.memory,power.draw,temperature.gpu,clocks.current.sm,clocks.current.memory --format=csv -l 0.05 > "$metrics_file" &
    NVIDIA_SMI_PID=$!
    
    # Sleep briefly to ensure nvidia-smi starts collecting before the workload
    sleep 1
    
    # Record the start time
    echo "Start time: $(date +%s.%N)" > "$output_file"
    
    # Modify opt_2dhisto.cu to use the specific strategy
    cp opt_2dhisto.cu opt_2dhisto.cu.backup
    
    # Comment out all kernel calls
    sed -i 's/^[ \t]*\(.*ParallelStrategy[A-Za-z]*<<<\)/\/\/\1/g' opt_2dhisto.cu
    sed -i 's/^[ \t]*\(.*Baseline<<<\)/\/\/\1/g' opt_2dhisto.cu
    
    # Uncomment the specific kernel strategy (using strategy 9 as it appears to be the latest)
    case $strategy in
        9)
            # Uncomment ParallelStrategyNine kernel
            sed -i '/\/\/\/\/\/\/ --- ParallelStrategyNine kernel ---/,/\/\/\/\/\/\/ --- ParallelStrategyNine kernel ---/s/\/\/\(.*ParallelStrategyNine<<<.*\)/\1/g' opt_2dhisto.cu
            ;;
        # Add other cases if needed
    esac
    
    # Rebuild and run the test_harness with the given seed
    make clean >> "$output_file" 2>&1
    make >> "$output_file" 2>&1
    ./test_harness $seed >> "$output_file" 2>&1
    
    # Record the end time
    echo "End time: $(date +%s.%N)" >> "$output_file"
    
    # Restore the original file
    mv opt_2dhisto.cu.backup opt_2dhisto.cu
    
    # Kill the nvidia-smi process
    kill $NVIDIA_SMI_PID
    
    # Extract important metrics from the output
    echo "Seed $seed with Strategy $strategy:" >> "$DATA_DIR/seed_results.txt"
    grep -A 2 "opt_2dhisto" "$output_file" >> "$DATA_DIR/seed_results.txt"
    grep "Test" "$output_file" >> "$DATA_DIR/seed_results.txt"
    echo "------------------------------------------------------" >> "$DATA_DIR/seed_results.txt"
    
    # Wait for a cooldown period between experiments
    echo "Cooldown period..."
    sleep 5
}

# Main script execution
echo "Starting histogram workload power data collection..."

# Part 1: Run with different kernel strategies
echo "Running experiments with different kernel strategies..."
for strategy in baseline 1 2 3 4 5 6 7 8 9; do
    run_kernel_experiment $strategy
done

# Part 2: Run with different seeds using the best strategy (strategy 9)
echo "Running experiments with different seeds..."
for seed in 0 42 100 500 1000 1500 2000; do
    run_seed_experiment $seed 9
done

# Process the collected data to create a summary
echo "Processing collected data..."

# Create a combined data file with experiment metadata for kernel strategies
{
    echo "Strategy,Start_Time,End_Time,Execution_Time_ms,Avg_GPU_Util,Avg_Mem_Util,Avg_Power_Draw,Max_Power_Draw,Avg_Temp,Avg_SM_Clock,Avg_Mem_Clock,Execution_Time_us"
    
    for strategy in baseline 1 2 3 4 5 6 7 8 9; do
        metrics_file="$DATA_DIR/histogram_metrics_strategy_${strategy}.csv"
        output_file="$DATA_DIR/histogram_power_strategy_${strategy}.csv"
        
        # Extract timestamps
        start_time=$(grep "Start time:" "$output_file" | cut -d' ' -f3)
        end_time=$(grep "End time:" "$output_file" | cut -d' ' -f3)
        
        # Calculate execution time in ms
        execution_time=$(echo "($end_time - $start_time) * 1000" | bc)
        
        # Extract metrics during the execution window
        avg_gpu_util=$(tail -n +2 "$metrics_file" | awk -F, '{sum+=$2} END {print sum/NR}')
        avg_mem_util=$(tail -n +2 "$metrics_file" | awk -F, '{sum+=$3} END {print sum/NR}')
        avg_power=$(tail -n +2 "$metrics_file" | awk -F, '{sum+=$4} END {print sum/NR}')
        max_power=$(tail -n +2 "$metrics_file" | awk -F, '{if($4>max) max=$4} END {print max}')
        avg_temp=$(tail -n +2 "$metrics_file" | awk -F, '{sum+=$5} END {print sum/NR}')
        avg_sm_clock=$(tail -n +2 "$metrics_file" | awk -F, '{sum+=$6} END {print sum/NR}')
        avg_mem_clock=$(tail -n +2 "$metrics_file" | awk -F, '{sum+=$7} END {print sum/NR}')
        
        # Extract algorithm execution time from the output (microseconds)
        exec_time=$(grep -A 2 "opt_2dhisto" "$output_file" | tail -n 1 | awk '{print $5}')
        
        echo "$strategy,$start_time,$end_time,$execution_time,$avg_gpu_util,$avg_mem_util,$avg_power,$max_power,$avg_temp,$avg_sm_clock,$avg_mem_clock,$exec_time"
    done
} > "$DATA_DIR/histogram_strategy_summary.csv"

# Create a combined data file with experiment metadata for seed experiments
{
    echo "Seed,Start_Time,End_Time,Execution_Time_ms,Avg_GPU_Util,Avg_Mem_Util,Avg_Power_Draw,Max_Power_Draw,Avg_Temp,Avg_SM_Clock,Avg_Mem_Clock,Execution_Time_us"
    
    for seed in 0 42 100 500 1000 1500 2000; do
        strategy=9  # Using strategy 9 for all seed tests
        metrics_file="$DATA_DIR/histogram_metrics_seed_${seed}_strategy_${strategy}.csv"
        output_file="$DATA_DIR/histogram_power_seed_${seed}_strategy_${strategy}.csv"
        
        # Extract timestamps
        start_time=$(grep "Start time:" "$output_file" | cut -d' ' -f3)
        end_time=$(grep "End time:" "$output_file" | cut -d' ' -f3)
        
        # Calculate execution time in ms
        execution_time=$(echo "($end_time - $start_time) * 1000" | bc)
        
        # Extract metrics during the execution window
        avg_gpu_util=$(tail -n +2 "$metrics_file" | awk -F, '{sum+=$2} END {print sum/NR}')
        avg_mem_util=$(tail -n +2 "$metrics_file" | awk -F, '{sum+=$3} END {print sum/NR}')
        avg_power=$(tail -n +2 "$metrics_file" | awk -F, '{sum+=$4} END {print sum/NR}')
        max_power=$(tail -n +2 "$metrics_file" | awk -F, '{if($4>max) max=$4} END {print max}')
        avg_temp=$(tail -n +2 "$metrics_file" | awk -F, '{sum+=$5} END {print sum/NR}')
        avg_sm_clock=$(tail -n +2 "$metrics_file" | awk -F, '{sum+=$6} END {print sum/NR}')
        avg_mem_clock=$(tail -n +2 "$metrics_file" | awk -F, '{sum+=$7} END {print sum/NR}')
        
        # Extract algorithm execution time from the output (microseconds)
        exec_time=$(grep -A 2 "opt_2dhisto" "$output_file" | tail -n 1 | awk '{print $5}')
        
        echo "$seed,$start_time,$end_time,$execution_time,$avg_gpu_util,$avg_mem_util,$avg_power,$max_power,$avg_temp,$avg_sm_clock,$avg_mem_clock,$exec_time"
    done
} > "$DATA_DIR/histogram_seed_summary.csv"

echo "Data collection complete. Results saved in $DATA_DIR"
