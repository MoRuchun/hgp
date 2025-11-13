<<<<<<< HEAD
# hgp
=======
# Hierarchical Gaussian Process Imputation

A Python framework for evaluating hierarchical Gaussian process regression in civil engineering experimental data with missing values.

## Overview

This project implements a chained multiple imputation system that supports both ordinary and hierarchical regression models (Linear Regression and Gaussian Process). It's designed to handle multi-source data with missing values and group-specific variations.

## Project Structure

```
.
├── models/                          # Regression models
│   ├── base_model.py               # Abstract base class
│   ├── linear_regression.py        # Linear regression (ordinary & hierarchical)
│   └── gaussian_process.py         # Gaussian process (ordinary & hierarchical)
├── imputation/                      # Imputation algorithms
│   └── chained_imputer.py          # Chained multiple imputation (MICE)
├── experiments/                     # Jupyter notebooks
│   ├── experiment_1_direct_imputation.ipynb
│   ├── experiment_2_downstream_impact.ipynb
│   └── experiment_3_unknown_group.ipynb
└── requirements.txt                 # Dependencies

```

## Installation

1. Clone the repository
2. Install dependencies:

```bash
pip install -r requirements.txt
```

## Usage

### Basic Example

```python
from models.linear_regression import LinearRegression
from models.gaussian_process import GaussianProcessRegression
from imputation.chained_imputer import ChainedImputer
import pandas as pd

# Load your data with missing values
data = pd.read_csv('your_data.csv')

# Create imputer with hierarchical linear regression
imputer = ChainedImputer(
    base_model=LinearRegression(),
    random_effects=['group_column'],  # Specify grouping variable
    n_imputations=5,
    max_iter=10
)

# Perform multiple imputation
imputed_datasets = imputer.fit_transform(data)

# Use the first imputed dataset
data_imputed = imputed_datasets[0]
```

### Experiments

Three Jupyter notebooks demonstrate the framework:

1. **Experiment 1: Direct Imputation Comparison**
   - Compares imputation accuracy across four methods
   - Evaluates MSE on artificially masked data

2. **Experiment 2: Downstream Model Impact**
   - Trains SVR models on imputed datasets
   - Measures how imputation quality affects prediction performance

3. **Experiment 3: Unknown Group Prediction**
   - Tests hierarchical models on unseen groups
   - Evaluates generalization capability

Run experiments:

```bash
cd experiments
jupyter notebook
```

## Models

### Linear Regression
- **Ordinary mode**: Bayesian Ridge regression
- **Hierarchical mode**: Mixed linear model with random effects

### Gaussian Process
- **Ordinary mode**: Standard GP regression
- **Hierarchical mode**: GP with group-specific random intercepts

## Key Features

- Flexible base model interface (easily extensible)
- Support for hierarchical structures via random effects
- Multiple imputation for uncertainty quantification
- Convergence checking in chained imputation
- Handles unknown groups in hierarchical models

## Requirements

- Python >= 3.8
- numpy >= 1.21.0
- pandas >= 1.3.0
- scikit-learn >= 1.0.0
- statsmodels >= 0.13.0
- GPy >= 1.10.0
- matplotlib >= 3.4.0
- seaborn >= 0.11.0
- jupyter >= 1.0.0

## License

MIT License

## Citation

If you use this code in your research, please cite:

```
[Your citation information here]
```
>>>>>>> 7bbe1bc (初始化项目)
