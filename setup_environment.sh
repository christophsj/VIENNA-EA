#!/bin/bash

# Setup script for PRASE-Python experiment environment
# This script sets up the conda environment on a new machine

set -e  # Exit on error

echo "========================================="
echo "PRASE-Python Environment Setup"
echo "========================================="

# Check if conda is installed
if ! command -v conda &> /dev/null; then
    echo "ERROR: conda is not installed or not in PATH"
    echo "Please install Miniconda or Anaconda first"
    exit 1
fi

echo "✓ conda found: $(which conda)"

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
ENV_FILE="$SCRIPT_DIR/environments/bert_int_environment.yml"

# Check if environment file exists
if [ ! -f "$ENV_FILE" ]; then
    echo "ERROR: Environment file not found at $ENV_FILE"
    exit 1
fi

echo "✓ Environment file found: $ENV_FILE"

# Extract environment name from yml file
ENV_NAME=$(grep "^name:" "$ENV_FILE" | awk '{print $2}')
echo "Environment name: $ENV_NAME"

# Check if environment already exists
if conda env list | grep -q "^$ENV_NAME "; then
    echo ""
    echo "WARNING: Environment '$ENV_NAME' already exists"
    read -p "Do you want to remove and recreate it? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "Removing existing environment..."
        conda env remove -n "$ENV_NAME" -y
    else
        echo "Keeping existing environment. Exiting."
        exit 0
    fi
fi

# Create conda environment from yml file
echo ""
echo "Creating conda environment from $ENV_FILE..."
echo "This may take several minutes..."
conda env create -f "$ENV_FILE"

echo ""
echo "========================================="
echo "✓ Environment setup complete!"
echo "========================================="
echo ""
echo "To activate the environment, run:"
echo "  conda activate $ENV_NAME"
echo ""
echo "After activation, you can run experiments using:"
echo "  ./run_experiments.sh"
echo ""
