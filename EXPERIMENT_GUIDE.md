# Experiment Setup and Execution Guide

This guide explains how to set up the environment and run comprehensive experiments on another machine.

## Overview

The experiment setup includes:
- **Learning Module**: BERT-INT
- **Alignment Module**: OverridedModule
- **Datasets**: 7 datasets (DBP15K variants and Wikidata variants)
- **Runs per dataset**: 3 (to collect mean and variance statistics)
- **PARIS iterations**: 10

## Files

- `setup_environment.sh` - Sets up the conda environment on a new machine
- `run_experiments.sh` - Runs all experiments across all datasets
- `environments/bert_int_environment.yml` - Conda environment specification

## Quick Start

### 1. Initial Setup (One-time)

On a new machine, run the environment setup script:

```bash
./setup_environment.sh
```

This will:
- Check if conda is installed
- Create a new conda environment called `base` from `environments/bert_int_environment.yml`
- Install all required dependencies (PyTorch, transformers, CUDA libraries, etc.)

**Note**: This may take 10-30 minutes depending on your internet connection.

### 2. Activate Environment

After setup, activate the environment:

```bash
conda activate base
```

### 3. Run Experiments

Execute all experiments:

```bash
./run_experiments.sh
```

This will run experiments on all 7 datasets, 3 times each (21 total runs).

## Datasets and Configuration

The following datasets will be tested with their corresponding description dictionaries:

| Dataset | Description Dictionary | Notes |
|---------|----------------------|-------|
| `dbp15k/fr_en/converted` | `data/des_dict.pkl` | French-English DBPedia |
| `dbp15k/ja_en/converted` | `data/des_dict.pkl` | Japanese-English DBPedia |
| `dbp15k/zh_en/converted` | `data/des_dict.pkl` | Chinese-English DBPedia |
| `D_W_15K_V2` | `data/D_W_15K_V2/des_dict_wd_15k_v2.pkl` | DBPedia-Wikidata |
| `D_Y_15K_V2` | None | DBPedia-YAGO |
| `EN_DE_15K_V2` | None | English-German |
| `EN_FR_15K_V2` | None | English-French |

## Experiment Output

### During Execution

The script will:
- Create a timestamped results directory: `experiment_results_YYYYMMDD_HHMMSS/`
- Generate a master log file: `experiment_log.txt`
- Create individual log files for each run: `{dataset}_run{1-3}.log`

### After Completion

Results will be organized in:
```
experiment_results_YYYYMMDD_HHMMSS/
├── experiment_log.txt
├── dbp15k_fr_en_converted_run1.log
├── dbp15k_fr_en_converted_run2.log
├── dbp15k_fr_en_converted_run3.log
├── dbp15k_ja_en_converted_run1.log
├── ...
└── EN_FR_15K_V2_run3.log

output/
├── dbp15k/fr_en/converted/
│   ├── bertint/
│   └── mapping/
│       └── PRASE-BertIntModule@*.txt
├── D_W_15K_V2/
│   ├── bertint/
│   └── mapping/
│       └── PRASE-BertIntModule@*.txt
└── ...
```

## Analyzing Results

### Extract Performance Metrics

Each experiment log contains performance metrics. To analyze results:

1. **Check individual run logs** in the `experiment_results_*/` directory
2. **Check mapping results** in `output/{dataset}/mapping/` directories
3. **Look for metrics** like Hits@1, Hits@10, MRR in the log files

### Computing Mean and Variance

For each dataset, you'll have 3 runs. Extract metrics from each run and compute:
- **Mean**: Average across 3 runs
- **Standard Deviation**: Variation across runs

Example metrics to track:
- Hits@1 (precision at rank 1)
- Hits@10 (precision at rank 10)
- MRR (Mean Reciprocal Rank)

## Command-Line Parameters

The experiments use the following parameters:

```bash
python test.py \
    --iterations 10 \
    --dataset_path {DATASET} \
    --alignment_module OverridedModule \
    [--des_dict_path {DES_DICT}]
```

### Parameter Details

- `--iterations`: Number of PARIS iterations (default: 10)
- `--dataset_path`: Path to dataset relative to `data/` directory
- `--alignment_module`: Alignment strategy (using `OverridedModule`)
- `--des_dict_path`: Path to description dictionary (optional, dataset-specific)
- `--interaction_model`: Boolean for BERT interaction model (default: True)
- `--training_max_percentage`: Max training data percentage (default: 0.2)

## Troubleshooting

### Environment Creation Fails

If conda environment creation fails:
```bash
# Update conda first
conda update -n base -c defaults conda

# Try creating environment again
./setup_environment.sh
```

### CUDA/GPU Issues

The environment is configured for CUDA 12. If you have a different CUDA version:
1. Check your CUDA version: `nvcc --version`
2. Modify `environments/bert_int_environment.yml` to match your CUDA version
3. Re-run `./setup_environment.sh`

### Out of Memory

If you encounter OOM errors:
- Reduce batch size in BERT-INT configuration
- Use a machine with more GPU memory
- Enable gradient checkpointing if available

### Missing Description Dictionaries

Some datasets may not have description dictionaries. This is normal - the script will run without them.

## Customization

### Running Specific Datasets

To run only specific datasets, edit `run_experiments.sh` and comment out datasets you don't want:

```bash
datasets=(
    "dbp15k/fr_en/converted|data/des_dict.pkl"
    # "dbp15k/ja_en/converted|data/des_dict.pkl"  # Commented out
    # ...
)
```

### Changing Number of Runs

To change from 3 runs to a different number, edit the loop in `run_experiments.sh`:

```bash
for run in {1..5}; do  # Changed from {1..3}
    run_experiment "$dataset_path" "$des_dict_path" "$run"
done
```

### Changing Alignment Module

To use a different alignment module, modify the `--alignment_module` parameter:

Available options:
- `OverridedModule` (current)
- `MergeAlignmentsModule`
- `OnlyAddUnalignedModule`
- `HigherConfidenceModule`

## Time Estimates

Approximate runtime per dataset (varies by size and hardware):
- **DBP15K datasets**: 30-60 minutes per run
- **Wikidata variants**: 45-90 minutes per run

**Total estimated time**: 15-30 hours for all 21 runs (7 datasets × 3 runs)

## Requirements

### System Requirements
- Linux OS (Ubuntu recommended)
- CUDA-capable GPU (CUDA 12 compatible)
- At least 16GB RAM
- At least 50GB free disk space

### Software Requirements
- Conda (Miniconda or Anaconda)
- CUDA 12 drivers
- Python 3.12 (installed via conda)

## Support

For issues related to:
- **Environment setup**: Check conda installation and CUDA drivers
- **Dataset paths**: Verify data files exist in `data/` directory
- **Memory issues**: Reduce batch sizes or use a larger GPU
- **Algorithm parameters**: Refer to the PRASE paper and test.py source code
