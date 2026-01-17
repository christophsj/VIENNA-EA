# Quick Reference Card - PRASE-Python Experiments

## Setup on New Machine

```bash
# 1. Clone repository and navigate to it
cd PRASE-Python

# 2. Run setup script
./setup_environment.sh

# 3. Activate environment
conda activate cluster
```

## Run All Experiments

```bash
# Run all datasets 3x each (21 total runs)
./run_experiments.sh
```

Expected runtime: **15-30 hours** total

## Analyze Results

```bash
# After experiments complete
python analyze_results.py experiment_results_YYYYMMDD_HHMMSS/
```

This generates:
- Console summary with mean ± std for each dataset
- `analysis_summary.json` with detailed statistics

## Manual Single Run

```bash
# Template
python test.py \
    --iterations 10 \
    --dataset_path <DATASET> \
    --alignment_module OverridedModule \
    [--des_dict_path <DES_DICT>]

# Examples
python test.py --iterations 10 --dataset_path dbp15k/fr_en/converted \
    --alignment_module OverridedModule --des_dict_path data/des_dict.pkl

python test.py --iterations 10 --dataset_path D_W_15K_V2 \
    --alignment_module OverridedModule \
    --des_dict_path data/D_W_15K_V2/des_dict_wd_15k_v2.pkl
```

## Datasets Overview

| Dataset | Description | Des Dict |
|---------|-------------|----------|
| `dbp15k/fr_en/converted` | FR-EN DBPedia | ✓ `data/des_dict.pkl` |
| `dbp15k/ja_en/converted` | JA-EN DBPedia | ✓ `data/des_dict.pkl` |
| `dbp15k/zh_en/converted` | ZH-EN DBPedia | ✓ `data/des_dict.pkl` |
| `D_W_15K_V2` | DBP-Wikidata | ✓ `data/D_W_15K_V2/des_dict_wd_15k_v2.pkl` |
| `D_Y_15K_V2` | DBP-YAGO | ✗ None |
| `EN_DE_15K_V2` | EN-DE | ✗ None |
| `EN_FR_15K_V2` | EN-FR | ✗ None |

## Configuration Used

- **Learning Module**: `BertIntModule` (BERT-INT)
- **Alignment Module**: `OverridedModule`
- **PARIS Iterations**: 10
- **Runs per Dataset**: 3

## Output Locations

```
experiment_results_YYYYMMDD_HHMMSS/    # Timestamped results directory
├── experiment_log.txt                  # Master log
├── {dataset}_run{1-3}.log             # Individual run logs
└── analysis_summary.json              # Analysis results (after running analyze_results.py)

output/{dataset}/                       # Model outputs
├── bertint/                           # BERT-INT intermediate outputs
└── mapping/                           # Final alignment mappings
    └── PRASE-BertIntModule@*.txt
```

## Troubleshooting

| Issue | Solution |
|-------|----------|
| `conda: command not found` | Install Miniconda/Anaconda |
| CUDA version mismatch | Edit `environments/cluster_env.yml` for your CUDA version |
| Out of memory | Use machine with more GPU RAM |
| Missing des_dict file | Normal for some datasets - will run without it |

## Key Metrics

Look for these in the logs:
- **Hits@1**: Top-1 accuracy
- **Hits@10**: Top-10 accuracy  
- **MRR**: Mean Reciprocal Rank
- **Precision/Recall**: Alignment quality

## Customization

### Run fewer datasets:
Edit `run_experiments.sh`, comment out datasets in the `datasets=()` array

### Change runs per dataset:
Edit `run_experiments.sh`, change `for run in {1..3}` to desired number

### Different alignment module:
Change `--alignment_module OverridedModule` to:
- `MergeAlignmentsModule`
- `OnlyAddUnalignedModule`
- `HigherConfidenceModule`

## Files Created

| File | Purpose |
|------|---------|
| `setup_environment.sh` | One-time environment setup |
| `run_experiments.sh` | Run all experiments |
| `analyze_results.py` | Compute statistics from results |
| `EXPERIMENT_GUIDE.md` | Detailed documentation |
| `QUICK_REFERENCE.md` | This file |

## Support

See `EXPERIMENT_GUIDE.md` for detailed troubleshooting and configuration options.
