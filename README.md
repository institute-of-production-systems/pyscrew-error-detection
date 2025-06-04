# Multi-Dataset Error Detection in Industrial Screw Driving Operations

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![MLflow](https://img.shields.io/badge/MLflow-tracking-green.svg)](https://mlflow.org/)
[![Work in Progress](https://img.shields.io/badge/Status-WIP-orange.svg)](https://github.com/Institute-of-Production-Systems/pyscrew-error-detection)
[![Research](https://img.shields.io/badge/Type-Research-purple.svg)](https://ips.mb.tu-dortmund.de/)

This repository provides a comprehensive machine learning framework for supervised error detection across all [PyScrew](https://github.com/nikolaiwest/pyscrew) datasets (s01-s06). Building upon our established methodology from the [ITISE 2025 paper](https://github.com/nikolaiwest/2025-supervised-error-detection-itise), this project extends the analysis to cover the complete spectrum of industrial screw driving scenarios and error types documented in the PyScrew dataset collection.

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

## 🏗️ Project Structure

```
├── data/                     # Data storage (downloaded via pyscrew)
├── mlruns/                   # MLflow experiment tracking
├── results/                  # Experiment results and visualizations
├── scripts/                  # Utility and plotting scripts
├── src/                      # Source code
│   ├── analysis/             # Analysis modules (legacy)
│   ├── data/                 # Data loading and preprocessing
│   │   ├── load.py          # PyScrew data interface
│   │   └── process.py       # PAA and normalization
│   ├── evaluation/           # Metrics and result containers
│   │   ├── apply_metrics.py # Sklearn metrics wrapper
│   │   └── results/         # 4-level result hierarchy
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

### Experiment Types

Our framework supports four complementary approaches:

| Experiment | Description | Use Case |
|------------|-------------|----------|
| `binary_vs_ref` | Each error class vs its own normal samples (50 vs 50) | **Balanced comparison** for detecting specific error types |
| `binary_vs_all` | Each error class vs ALL normal samples (50 vs 1215) | **Realistic imbalanced** scenario for anomaly detection |
| `multiclass_with_groups` | Multi-class within error groups (5 groups) | **Grouped classification** of related error mechanisms |
| `multiclass_with_all` | All 25 error classes + normal (26-class problem) | **Comprehensive classification** across all error types |

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
python main.py --experiment binary_vs_ref --models fast
```

**Run with more cross-validation folds:**
```bash
python main.py --cv-folds 10
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
    experiment_name="binary_vs_ref",
    model_selection="paper",
    scenario_id="s04",           # Configurable dataset
    target_length=2000,          # Fixed preprocessing  
    cv_folds=5,                  # Configurable via CLI
    random_seed=42,              # Configurable via CLI
    n_jobs=-1                    # Use all cores
)

results = runner.run()
```

**Available CLI options:**
```bash
python main.py --help

# Core options:
--experiment {binary_vs_ref,binary_vs_all,multiclass_with_groups,multiclass_with_all,all}
--models {debug,fast,paper,full,sklearn,sktime}
--cv-folds N
--random-seed N
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

### Cross-Validation Strategy

- **5-fold stratified CV** (default)
- **Stratification** by class to handle imbalance
- **Reproducible** with fixed random seeds
- **Parallel execution** across models and folds

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

> 💡 **Tip**: Start with `python main.py --experiment binary_vs_ref --models fast` for a quick overview, then explore the MLflow UI at `http://localhost:5000` to dive into detailed results!