# Multi-Dataset Error Detection in Industrial Screw Driving Operations

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![MLflow](https://img.shields.io/badge/MLflow-tracking-green.svg)](https://mlflow.org/)
[![Work in Progress](https://img.shields.io/badge/Status-WIP-orange.svg)](https://github.com/Institute-of-Production-Systems/pyscrew-error-detection)
[![Research](https://img.shields.io/badge/Type-Research-purple.svg)](https://ips.mb.tu-dortmund.de/)

This repository provides a comprehensive machine learning framework for supervised error detection across all [PyScrew](https://github.com/nikolaiwest/pyscrew) datasets (s01-s06). Building upon our established methodology from the [ITISE 2025 paper](https://github.com/nikolaiwest/2025-supervised-error-detection-itise), this project extends the analysis to cover the complete spectrum of industrial screw driving scenarios and error types documented in the PyScrew dataset collection.

> ⚠️ **Note:** This project is under active development. Features may change or break.

## Overview

Industrial screw driving operations are critical in manufacturing processes, yet prone to various failure modes that can compromise product quality and production efficiency. This framework implements multi-class error detection algorithms capable of identifying and classifying 25+ distinct error types across different assembly conditions, screw materials, and operational parameters.

**Key Features:**
- **Multi-dataset compatibility**: Supports all PyScrew scenarios (s01-s06)
- **Comprehensive error taxonomy**: Handles diverse failure modes from thread deformation to torque anomalies
- **Scalable ML pipeline**: Automated preprocessing, feature extraction, and model training
- **Reproducible experiments**: Standardized evaluation protocols with cross-validation
- **Industrial applicability**: Real-world error patterns from production environments

## Research Foundation

This work extends our previous research on supervised error detection in industrial screw driving operations, originally presented at ITISE 2025. The framework has been generalized to accommodate the full range of PyScrew datasets, enabling comparative analysis across different operational scenarios and assembly conditions.

### Publications 
- **West, N., & Deuse, J.** (2025). PyScrew: A Comprehensive Dataset Collection from Industrial Screw Driving Experiments. *arXiv preprint*. [![arXiv](https://img.shields.io/badge/arXiv-2505.11925-b31b1b.svg)](https://arxiv.org/abs/2505.11925)

- **West, N., Klute, M., Heim, H.-P., & Deuse, J.** (2025). Cross-Process Error Detection in Industrial Screw Driving Operations Using Time Series Classification. *Proceedings of the 7th World Symposium on Software Engineering (WSSE '25)*, Okayama, Japan. ACM. [![DOI](https://img.shields.io/badge/ACM-979--8--4007--1577--8-blue.svg)](http://wsse2025.sciei.org/)

## 🏗️ Project Structure

```
├── data/                     # Data storage (downloaded via pyscrew)
├── mlruns/                   # MLflow experiment tracking
├── results/                  # Experiment results and visualizations
├── scripts/                  # Utility and plotting scripts
├── src/                      # Source code
│   ├── analysis/             # Analysis modules (legacy)
│   ├── data/                 # Data loading and preprocessing
│   │   ├── load.py           # PyScrew data interface
│   │   └── process.py        # PAA and normalization
│   ├── evaluation/           # Metrics and result containers
│   │   ├── get_metrics.py    # Sklearn metrics wrapper
│   │   └── results/          # 4-level result hierarchy
│   ├── experiments/          # Experiment orchestration
│   │   ├── experiment_runner.py  # Main experiment runner
│   │   ├── sampling.py           # Dataset generation strategies
│   │   └── training.py           # Cross-validation training
│   ├── mlflow/               # MLflow integration
│   │   ├── manager.py        # Hierarchical logging manager
│   │   └── server.py         # Server management
│   ├── models/               # Model configurations
│   │   ├── classifiers.py    # Dynamic model loading
│   │   ├── sklearn_models.yml    # Sklearn model configs
│   │   └── sktime_models.yml     # Sktime model configs
│   ├── plots/                # Visualization modules
│   └── utils/                # Logging and utilities
├── .gitignore
├── LICENSE                   # MIT License
├── main.py                   # Main entry point
├── README.md                 # This file
├── requirements.txt          # Python dependencies
└── setup.py                  # Package installation
```

## 🚀 Quick Start

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Institute-of-Production-Systems/pyscrew-error-detection.git
   cd 2025-supervised-error-detection-itise
   ```

2. **Create and activate virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

### Quick Dataset Access

If you just want to explore the dataset *(or refer to our library [Pyscrew](https://github.com/nikolaiwest/pyscrew) for more info on the data)*:

```python
import pyscrew
data = pyscrew.get_data("s04")  # Downloads ~2GB of time series data
print(f"Dataset contains {len(data['torque_values'])} samples")
print(f"Classes: {set(data['class_values'])}")
```

## 🧪 Experiments

### s04 Sampling 

Our framework supports four complementary approaches (applicable for s04):

| Experiment | Description | Use Case |
|------------|-------------|----------|
| `binary_vs_ref` | Each error class vs its own normal samples (50 vs 50) | **Balanced comparison** for detecting specific error types |
| `binary_vs_all` | Each error class vs ALL normal samples (50 vs 1215) | **Realistic imbalanced** scenario for anomaly detection |
| `multiclass_with_groups` | Multi-class within error groups (5 groups) | **Grouped classification** of related error mechanisms |
| `multiclass_with_all` | All 25 error classes + normal (26-class problem) | **Comprehensive classification** across all error types |

### s05 + s06 Sampling
The s05 and s06 sampling frameworks use an elegant **2×2 matrix approach** based on two key decisions:

| | **All Classes** | **Only Strongest Variations** |
|---|---|---|
| **Combined Groups** | `combined_all_classes`: Single dataset with normal operation vs all parameter variations. Explores with anormal parameter variations can be distinguished in a large group. | `combined_only_extreme`: Single dataset with normal + most extreme from each parameter group (6 classes). Focuses on the extreme deviations only to see if there is an effect at all. |
| **Separate Groups** | `separate_all_classes`: Five separate datasets, each classifying all parameter values within one group. Allows to see if (which) variations in the parameter range are notable. | `separate_only_extreme`: Five separate binary datasets (normal vs most extreme per group). Should serve as a dummy-test to compare if the variation (in its most extreme form) is even detectable. |

### Model Selection Options

| Selection | Models | Use Case |
|-----------|---------|----------|
| `debug` | DummyClassifier only | Quick testing |
| `fast` | 3 fast models | Rapid prototyping |
| `paper` | 5 representative models | **Paper results** (recommended) |
| `full` | 15+ comprehensive models | Exhaustive comparison |
| `sklearn` | Sklearn models only | Traditional ML focus |
| `sktime` | Time series models only | Specialized TS methods |

### Running Experiments

**Run all experiments (paper setup):**
```bash
python main.py
```

**Run specific experiment:**
```bash
python main.py --scenario s04 --sampling binary_vs_ref --modeling fast
```

**Quiet mode for production:**
```bash
python main.py --quiet
```

### Advanced Usage

**Custom experiment runner (if you want to change more things):**
```python
from src.experiments import ExperimentRunner

runner = ExperimentRunner(
    scenario_selection="s04",           # Configurable dataset
    sampling_selection="binary_vs_ref",
    modeling_selection="paper",
    cv_folds=5,                  # Configurable via config.yml
    random_seed=42,              # Configurable via config.yml
    fail_fast=False,             # Continue on failures vs stop immediately
    n_jobs=-1                    # Use all cores
)

results = runner.run()
```

**Available CLI options:**
```bash
python main.py --help

# Core options:
--scenario {s04,s05,s06}
--sampling {binary_vs_ref,binary_vs_all,multiclass_with_groups,multiclass_with_all,all}
--modeling {debug,fast,paper,full,sklearn,sktime}
--quiet / --verbose
```

## 📊 Results & Visualization

### MLflow Tracking

Every experiment automatically logs to MLflow with 4-level hierarchy:
- **Experiment** → Overall comparison across datasets
- **Dataset** → Model performance on specific error types  
- **Model** → Cross-validation results and stability metrics
- **Fold** → Individual CV fold results and confusion matrices

**Start MLflow UI:**
```bash
mlflow ui --port 5000
# Open http://localhost:5000
```

## 🔬 Technical Details

### Data Processing Pipeline

1. **Loading**: PyScrew interface to configurable dataset (s01-s06)
2. **PAA Reduction**: 2000 → 200 time points (configurable)  
3. **Normalization**: Z-score standardization per time series
4. **Sampling**: Generate datasets based on experiment type

### Model Architecture

- **Traditional ML**: Random Forest, SVM, Logistic Regression 
- **Time Series**: ROCKET, Time Series Forest, BOSS Ensemble
- **Advanced**: Elastic Ensemble, Shapelet Transform

### Robust Failure Handling

- **Hierarchical exception handling**: Each level (Experiment→Dataset→Model→Fold) handles failures gracefully
- **Configurable behavior**: `fail_fast=false` continues despite failures, `fail_fast=true` stops immediately  
- **Rich failure tracking**: CSV exports include failure flags at all levels for post-analysis
- **Production ready**: Experiments complete even with partial failures, providing maximum data for analysis

### Cross-Validation Strategy

- **5-fold stratified CV** (default)
- **Stratification** by class to handle imbalance
- **Reproducible** with fixed random seeds
- **Parallel execution** across models and folds

## 📊 MLflow Tracking Architecture

### 4-Level Hierarchy

Our framework implements a hierarchical tracking structure that mirrors the experimental design:

```
Experiment (e.g., "run_s04_binary_vs_ref_paper")
└── Dataset (e.g., "101_deformed-thread")
    └── Model (e.g., "RandomForestClassifier")
        └── Fold (e.g., "fold_0")
```

### Tag Inheritance

Tags propagate down the hierarchy, enabling us to do cross-experiment queries. Each level inherits tags from its parent and adds its own:

| Tag | Experiment | Dataset | Model | Fold | Description |
|-----|------------|---------|-------|------|-------------|
| **Core Configuration** |||||
| `scenario` | ✓ | ✓ | ✓ | ✓ | Dataset selection (s04/s05/s06) |
| `sampling` | ✓ | ✓ | ✓ | ✓ | Sampling strategy (binary_vs_ref, etc.) |
| `modeling` | ✓ | ✓ | ✓ | ✓ | Model set (debug/fast/paper/full) |
| **Timing** |||||
| `start_time` | ✓ | ✓ | ✓ | ✓ | Pipeline start timestamp |
| `end_time` | ✓ | ✓ | ✓ | ✓ | Pipeline end timestamp |
| `run_time` | ✓ | ✓ | ✓ | ✓ | Human-readable duration |
| **Dataset Metadata** |||||
| `n_samples` | | ✓ | ✓ | ✓ | Total samples in dataset |
| `n_classes` | | ✓ | ✓ | ✓ | Number of unique classes |
| `samples_per_class` | | ✓ | ✓ | ✓ | Class distribution (e.g., "120:45") |
| **Model Metadata** |||||
| `model_name` | | | ✓ | ✓ | Classifier name |
| `model_family` | | | ✓ | ✓ | Algorithm family (Tree, Neural, etc.) |
| `model_library` | | | ✓ | ✓ | sklearn or sktime |
| **Fold Metadata** |||||
| `fold_index` | | | | ✓ | CV fold number (0-4) |
| `n_train` | | | | ✓ | Training samples in fold |
| `n_test` | | | | ✓ | Test samples in fold |


## 📋 Requirements

- **Python**: 3.8+
- **Core**: scikit-learn, sktime, pandas, numpy
- **Visualization**: matplotlib, seaborn
- **Tracking**: MLflow
- **Data**: pyscrew (auto-downloads datasets)

See `requirements.txt` for complete dependencies.

## 🚧 Development Status

**Current Phase:** Multi-dataset integration and model optimization

**Completed:**
- ✅ Framework forked from ITISE 2025 methodology
   - ✅ MLflow tracking and result visualization
   - ✅ PyScrew s04 dataset integration and validation

**In Progress:**
- 🔄 Integration of PyScrew datasets s01, s02, s03, s05, s06
- 🔄 Cross-dataset performance analysis
- 🔄 Model hyperparameter optimization for diverse scenarios

**Planned:**
- 📋 Comparative analysis across all dataset scenarios
- 📋 Publication of comprehensive multi-dataset results
- 📋 Extended model evaluation framework

## 🤝 Contributing

We welcome contributions! Please see our [contribution guidelines](CONTRIBUTING.md) for details.

## 📞 Contact

- **Author**: Nikolai West
- **Email**: nikolai.west@tu-dortmund.de
- **Institution**: Technical University Dortmund, Institute for Production Systems
- **Project**: [prodata-projekt.de](https://prodata-projekt.de/)

## 🏛️ Acknowledgments

This research is supported by:

| Organization | Role | 
|-------------|------|
| **German Ministry of Education and Research (BMBF)** | Primary funding through "Data competencies for early career researchers" program |
| **European Union's NextGenerationEU** | Co-funding initiative |
| **VDIVDE Innovation + Technik GmbH** | Program administration and support |

**Research Partners:**
- [RIF Institute for Research and Transfer e.V.](https://www.rif-ev.de/) - Dataset collection and preparation
- [Technical University Dortmund - Institute for Production Systems](https://ips.mb.tu-dortmund.de/) - Research execution

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

> 💡 **Tip**: Start with `python main.py --scenario s04 --sampling binary_vs_ref --modeling fast` for a quick overview, then explore the MLflow UI at `http://localhost:5000` to dive into detailed results!