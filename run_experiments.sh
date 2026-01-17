#!/bin/bash

# Comprehensive experiment runner for PRASE-Python
# Runs each dataset 3 times to collect mean and variation results
# Uses BERT-INT as learning module and OverridedModule as alignment module

set -e  # Exit on error

echo "========================================="
echo "PRASE-Python Experiment Runner"
echo "========================================="
echo "Configuration:"
echo "  - Learning Module: BERT-INT"
echo "  - Alignment Module: OverridedModule"
echo "  - Iterations per dataset: 10"
echo "  - Runs per dataset: 3"
echo "========================================="
echo ""

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Create results directory with timestamp
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
RESULTS_DIR="$SCRIPT_DIR/experiment_results_$TIMESTAMP"
mkdir -p "$RESULTS_DIR"

# Log file
LOG_FILE="$RESULTS_DIR/experiment_log.txt"

# Function to log messages
log_message() {
    echo "$1" | tee -a "$LOG_FILE"
}

# Function to run experiment
run_experiment() {
    local dataset_path=$1
    local des_dict_path=$2
    local run_number=$3
    local dataset_name=$(basename "$dataset_path")
    
    log_message ""
    log_message "========================================="
    log_message "Dataset: $dataset_name"
    log_message "Run: $run_number/3"
    log_message "Des Dict: ${des_dict_path:-None}"
    log_message "========================================="
    
    # Build command
    local cmd="python test.py --iterations 10 --dataset_path $dataset_path --alignment_module OverridedModule"
    
    # Add des_dict_path if provided
    if [ -n "$des_dict_path" ]; then
        cmd="$cmd --des_dict_path $des_dict_path"
    fi
    
    log_message "Command: $cmd"
    log_message "Start time: $(date)"
    
    # Run the experiment and capture output
    local run_log="$RESULTS_DIR/${dataset_name}_run${run_number}.log"
    
    if eval "$cmd" 2>&1 | tee "$run_log"; then
        log_message "✓ Completed successfully"
    else
        log_message "✗ Failed with error code $?"
    fi
    
    log_message "End time: $(date)"
}

# Array of datasets with their des_dict paths
# Format: "dataset_path|des_dict_path"
datasets=(
    "dbp15k/fr_en/converted|data/des_dict.pkl"
    "dbp15k/ja_en/converted|data/des_dict.pkl"
    "dbp15k/zh_en/converted|data/des_dict.pkl"
    "D_W_15K_V2|data/D_W_15K_V2/des_dict_wd_15k_v2.pkl"
    "D_Y_15K_V2|"
    "EN_DE_15K_V2|"
    "EN_FR_15K_V2|"
)

# Start timing
EXPERIMENT_START=$(date +%s)
log_message "========================================="
log_message "Experiment started at: $(date)"
log_message "Results will be saved to: $RESULTS_DIR"
log_message "========================================="

# Run each dataset 3 times
for dataset_spec in "${datasets[@]}"; do
    # Split dataset_path and des_dict_path
    IFS='|' read -r dataset_path des_dict_path <<< "$dataset_spec"
    
    for run in {1..3}; do
        run_experiment "$dataset_path" "$des_dict_path" "$run"
        
        # Add a small delay between runs
        sleep 2
    done
done

# End timing
EXPERIMENT_END=$(date +%s)
DURATION=$((EXPERIMENT_END - EXPERIMENT_START))
HOURS=$((DURATION / 3600))
MINUTES=$(((DURATION % 3600) / 60))
SECONDS=$((DURATION % 60))

log_message ""
log_message "========================================="
log_message "All experiments completed!"
log_message "========================================="
log_message "Total duration: ${HOURS}h ${MINUTES}m ${SECONDS}s"
log_message "Results directory: $RESULTS_DIR"
log_message ""

# Summary of outputs
log_message "Output directories:"
log_message "  - Experiment logs: $RESULTS_DIR/*.log"
log_message "  - Model outputs: output/*"
log_message ""
log_message "To analyze results, check the following directories:"
for dataset_spec in "${datasets[@]}"; do
    IFS='|' read -r dataset_path _ <<< "$dataset_spec"
    dataset_name=$(basename "$dataset_path")
    log_message "  - output/$dataset_path/mapping/"
done

echo ""
echo "Experiment log saved to: $LOG_FILE"
