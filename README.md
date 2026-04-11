# VIENNA-EA

**VIENNA-EA** is an extended framework for Knowledge Graph Entity Alignment, built on top of [PRASE-Python](https://github.com/qizhyuan/PRASE-Python) (the code accompanying the IJCAI paper *"Unsupervised Knowledge Graph Alignment by Probabilistic Reasoning and Semantic Embedding"*).

This project extends the original PRASE pipeline with:
- A modular learning/alignment architecture (`module/`)
- **BERT-INT** as the primary learning module
- Pluggable **alignment modules** (Override, Merge, HigherConfidence, OnlyAddUnaligned)
- and more

---

## Project Structure

```
VIENNA-EA/
├── test.py                    # Main entry point
├── run_experiments.sh         # Automated multi-dataset experiment runner
├── analyze_results.py         # Post-experiment results analysis
├── setup_environment.sh       # Conda environment setup script
│
├── module/                    # Modular pipeline components
│   ├── module.py              # Abstract Module / AlignmentState definitions
│   ├── collection_utils.py
│   ├── alignment/             # Alignment strategy modules
│   │   ├── override_module.py
│   │   ├── merge_alignments_module.py
│   │   ├── higher_confidence_module.py
│   │   ├── only_add_unaligned_module.py
│   │   └── alignment_module_factory.py
│   └── learning/              # Learning (embedding) modules
│       ├── bert_int_module.py
│       ├── cluster_ea_module.py
│       ├── precomputed_embedding_module.py
│       └── dummy_module.py
│
├── model/                     # Model implementations
│   ├── paris/                 # PARIS probabilistic reasoning engine
│   ├── bert_int/              # BERT-INT embedding model
│   └── clusterea/             # ClusterEA embedding model
│
├── objects/                   # Core KG data structures
│   ├── Entity.py
│   ├── Relation.py
│   ├── KG.py
│   └── KGs.py
│
├── data/                      # Benchmark datasets
│   ├── dbp15k/                # DBP15K (fr_en, ja_en, zh_en)
│   ├── D_W_15K_V2/            # DBPedia–Wikidata
│   ├── D_Y_15K_V2/            # DBPedia–YAGO
│   ├── EN_DE_15K_V2/          # English–German
│   └── EN_FR_15K_V2/          # English–French
│
├── environments/
│   └── bert_int_environment.yml  # Conda environment spec
│
└── experiments/               # Stored experiment results
```

---

## Installation

### Prerequisites
- [Miniconda](https://docs.conda.io/en/latest/miniconda.html) or Anaconda
- CUDA-compatible GPU (recommended)

### Setup

```bash
# Clone the repository
git clone https://github.com/christophsj/VIENNA-EA.git
cd VIENNA-EA

# Create and activate the conda environment
./setup_environment.sh
conda activate bert
```

The setup script installs all required dependencies including PyTorch, Transformers, and CUDA libraries.

---

## Usage

### Single Run

```bash
python test.py \
    --iterations 10 \
    --dataset_path <DATASET> \
    --alignment_module OverridedModule \
    [--des_dict_path <DES_DICT_PATH>]
```

**Examples:**

```bash
# DBP15K French-English
python test.py --iterations 10 --dataset_path dbp15k/fr_en/converted \
    --alignment_module OverridedModule --des_dict_path data/des_dict.pkl

# DBPedia-Wikidata
python test.py --iterations 10 --dataset_path D_W_15K_V2 \
    --alignment_module OverridedModule \
    --des_dict_path data/D_W_15K_V2/des_dict_wd_15k_v2.pkl

# DBPedia-YAGO (no description dictionary)
python test.py --iterations 10 --dataset_path D_Y_15K_V2 \
    --alignment_module OverridedModule
```

**Arguments:**

| Argument | Default | Description |
|---|---|---|
| `--iterations` | `10` | Number of PARIS iterations |
| `--dataset_path` | `D_W_15K_V2/` | Path to dataset directory (relative to `data/`) |
| `--alignment_module` | `MergeAlignmentsModule` | Alignment strategy to use |
| `--des_dict_path` | `None` | Path to description dictionary (`.pkl`) |
| `--training_max_percentage` | `0.2` | Max fraction of data used for training |
| `--interaction_model` | `True` | Whether to use BERT-INT interaction model |
| `--model_path` | `None` | Path to a pre-trained model checkpoint |
| `--filter_entities` | `True` | Restrict KG to entities present in ground truth |

### Run All Experiments

```bash
./run_experiments.sh
```



### Analyze Results

```bash
python analyze_results.py experiment_results_YYYYMMDD_HHMMSS/
```

---

## Datasets

| Dataset | Description | Description Dict |
|---|---|---|
| `dbp15k/fr_en/converted` | French–English DBPedia | ✓ `data/des_dict.pkl` |
| `dbp15k/ja_en/converted` | Japanese–English DBPedia | ✓ `data/des_dict.pkl` |
| `dbp15k/zh_en/converted` | Chinese–English DBPedia | ✓ `data/des_dict.pkl` |
| `D_W_15K_V2` | DBPedia–Wikidata | ✓ `data/D_W_15K_V2/des_dict_wd_15k_v2.pkl` |
| `D_Y_15K_V2` | DBPedia–YAGO | ✗ |
| `EN_DE_15K_V2` | English–German | ✗ |
| `EN_FR_15K_V2` | English–French | ✗ |

---

## Alignment Modules

The alignment module controls how BERT-INT predictions are merged back into the PRASE alignment state:

| Module | Description |
|---|---|
| `OverridedModule` | New predictions fully override previous ones (recommended) |
| `MergeAlignmentsModule` | Probabilities are accumulated additively |
| `HigherConfidenceModule` | Keeps the higher-confidence alignment per entity |
| `OnlyAddUnalignedModule` | Only adds predictions for previously unaligned entities |

---

## Architecture

The pipeline follows an iterative loop:

1. **PARIS** runs probabilistic reasoning over relational/attribute triples to produce initial alignment probabilities.
2. The **learning module** (BERT-INT) uses current alignments as training signal and produces updated entity embeddings and alignment predictions.
3. The **alignment module** merges BERT-INT predictions back into the PRASE state.
4. Steps 1–3 repeat for the configured number of iterations.

A configurable **fusion function** balances the PARIS probability and the embedding cosine similarity:

$$\text{score}(e_1, e_2) = 0.8 \cdot p_{\text{PARIS}} + 0.2 \cdot \cos(\mathbf{e}_1, \mathbf{e}_2)$$

---

## Attribution

The PARIS/PRASE core is based on [PRASE-Python](https://github.com/qizhyuan/PRASE-Python) by qizhyuan (nju-websoft), which accompanies the paper:

> *Unsupervised Knowledge Graph Alignment by Probabilistic Reasoning and Semantic Embedding*, IJCAI 2021.

BERT-INT is integrated from the [BERT-INT](https://github.com/kosugi11037/bert-int) implementation.
