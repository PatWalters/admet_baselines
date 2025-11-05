# ADMET Baselines

Baseline machine learning models for ADMET (Absorption, Distribution, Metabolism, Excretion, and Toxicity) property prediction from molecular SMILES strings.

## Overview

This repository contains pre-trained baseline models for predicting:
- **CYP (Cytochrome P450) enzyme interactions**: Classification models for inhibition and substrate prediction across multiple CYP isoforms (1A2, 2C19, 2C9, 2D6, 3A4)
- **PXR (Pregnane X Receptor) activation**: Regression model for predicting EC50 values

## Features

- Direct SMILES string input (no manual fingerprint generation required)
- Handles invalid/unparsable SMILES gracefully
- Multiple CYP isoforms and interaction types supported
- Uncertainty quantification for PXR predictions
- Easy-to-use Python API

## Installation

1. Clone this repository:
```bash
git clone <repository-url>
cd admet_baselines
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Quick Start

### CYP Classification

```python
from cyp_classifier import CYPClassifier

# Load pre-trained models
classifier = CYPClassifier('CYP_classifier.pkl')

# See available models
print(classifier.available_model_ids())
# ['1A2_Inh', '2C19_Inh', '2C9_Inh', '2D6_Inh', '3A4_Inh', 
#  '1A2_Sub', '2C19_Sub', '2C9_Sub', '2D6_Sub', '3A4_Sub']

# Predict inhibition for CYP1A2
smiles_list = ['CCO', 'CC(=O)O', 'c1ccccc1']
predictions = classifier.predict('1A2_Inh', smiles_list)

# Get probabilities
probabilities, classes = classifier.predict_proba('2C9_Sub', smiles_list, return_classes=True)
```

### PXR Regression

```python
from pxr_regressor import PXRRegressor

# Load pre-trained model
regressor = PXRRegressor('PXR_regressor.pkl')

# Predict EC50 values with uncertainty
smiles_list = ['CCO', 'CC(=O)O', 'c1ccccc1']
predictions, uncertainties = regressor.predict(smiles_list)
```

## Model Details

### CYP Classifier

- **Model Type**: LightGBM classifier
- **Features**: Molecular fingerprints (via `useful_rdkit_utils`)
- **Models Available**:
  - Inhibition prediction: `1A2_Inh`, `2C19_Inh`, `2C9_Inh`, `2D6_Inh`, `3A4_Inh`
  - Substrate prediction: `1A2_Sub`, `2C19_Sub`, `2C9_Sub`, `2D6_Sub`, `3A4_Sub`
- **Output**: Class predictions and probabilities

### PXR Regressor

- **Model Type**: Gaussian Process Regressor with Tanimoto kernel
- **Features**: Molecular fingerprints (via `useful_rdkit_utils`)
- **Output**: EC50 predictions with uncertainty estimates

## Training

See the training notebooks for details on model training:
- `train_cyp_classifier.ipynb`: Training CYP classification models
- `train_pxr_regressor.ipynb`: Training PXR regression model

## Data Sources

- CYP data: Veith et al. dataset and Carbonmangels dataset
- PXR data: ChEMBL EC50 values

## File Structure

```
admet_baselines/
cyp_classifier.py           # CYP classification models
pxr_regressor.py            # PXR regression model
CYP_classifier.pkl          # Pre-trained CYP models
PXR_regressor.pkl           # Pre-trained PXR model
data/                       # Training data files
train_cyp_classifier.ipynb  # Training the CYP models
train_pxr_regressor.ipynb   # Training the PXR model
demo_models.ipynb           # Usage examples
requirements.txt            # Python dependencies
```

## API Reference

### CYPClassifier

```python
class CYPClassifier:
    def __init__(self, model_dict_or_filename: Union[Dict, str], smi2fp=None)
    def predict(self, key: str, smiles_list: List[str], suppress_warnings=True) -> np.ndarray
    def predict_proba(self, key: str, smiles_list: List[str], suppress_warnings=True, return_classes=False)
    def available_model_ids(self) -> List[str]
    def save(self, filename: str) -> None
```

### PXRRegressor

```python
class PXRRegressor:
    def __init__(self, model_file: Optional[str] = None)
    def fit(self, smiles_list: List[str], y_list: List[float]) -> 'PXRRegressor'
    def predict(self, smiles_list: List[str]) -> Tuple[List[float], List[float]]
    def save(self, model_file: str) -> None
```

## Requirements

- Python 3.11+
- numpy
- scikit-learn
- lightgbm
- rdkit
- useful-rdkit-utils
- pandas (for notebooks)
- matplotlib, seaborn (for notebooks)
- tdc (for data loading in notebooks)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

