# Uncertainty Quantification for Tabular Regression

> Decomposing **aleatoric** (data noise) and **epistemic** (model) uncertainty in vehicle value forecasting — with actionable signals for risk-aware decision support.

---

## Overview

Point predictions alone are not enough for high-stakes regression tasks. This project implements and evaluates a **modular uncertainty estimation stack** for tabular regression, focusing on whether decomposed uncertainty signals can reliably flag unreliable predictions — particularly under distribution shift and for high-error cases.

The pipeline was evaluated on a **public used-vehicle price dataset (Craigslist)**, providing a fully reproducible benchmark for the methods described here.

---

## Key Features

- **Heteroscedastic likelihood heads** — Gaussian and Laplace aleatoric uncertainty estimation
- **Normalizing Flow refinement** — conditional 1D flow trained on standardized residuals to capture non-Gaussian noise structure
- **MC Dropout** — lightweight epistemic uncertainty via stochastic inference passes
- **Deep Ensembles** — epistemic uncertainty from independently trained model members
- **DIDO** — Discretization-Induced Dirichlet Posterior as a Dirichlet-based epistemic ranking signal
- **Post-hoc calibration** — variance scaling fitted on a validation split for consistent interval diagnostics
- **kNN-based OOD construction** — out-of-distribution subsets derived from feature-space distance to the training distribution
- **Decision-oriented evaluation** — large-error detection AUC, OOD separation AUC, calibration coverage curves, and signal decoupling diagnostics

---

## Results Highlights (Craigslist dataset)

| Signal | Large-error Detection AUC |
|---|---|
| Aleatoric (Gaussian NF) | **~0.886** |
| Epistemic (Deep Ensemble) | ~0.840 |
| Epistemic (MC Dropout) | ~0.812 |
| Random baseline | 0.500 |

- Normalizing Flow refinement **consistently improves** distributional fit (CRPS) without degrading point accuracy (MAE)
- Deep Ensembles and DIDO **outperform MC Dropout** for OOD separation
- Aleatoric uncertainty captures the majority of large-error signal; blending with epistemic yields modest additional gains

---

## Tech Stack

| Component | Tool |
|---|---|
| Framework | PyTorch |
| Hyperparameter Optimization | Optuna |
| Normalizing Flows | Custom 1D conditional flow (affine / spline transforms) |
| Experiment tracking | YAML-based config manifests |
| Data | [Craigslist Cars+Trucks](https://www.kaggle.com/datasets/austinreese/craigslist-carstrucks-data) (Kaggle) |

---

## Repository Structure

```
uncertainty_quantification/
├── configs/              # YAML/JSON experiment configurations
├── scripts/              # Training, evaluation, and calibration code
├── notebooks/            # Analysis, plots, and result summaries
├── optuna_studies/       # Hyperparameter optimization runs
└── .gitignore
```

---

## Quickstart

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Download data

Download the [Craigslist Cars+Trucks dataset](https://www.kaggle.com/datasets/austinreese/craigslist-carstrucks-data) from Kaggle and place it in `data/raw/`.

### 3. Train a model

```bash
python scripts/train.py --config configs/gauss_nf.yaml
```

### 4. Run hyperparameter optimization

```bash
python scripts/hpo.py --config configs/hpo_gauss.yaml
```

### 5. Evaluate uncertainty signals

```bash
python scripts/evaluate.py --config configs/eval.yaml
```

---

## Methods in Brief

### Aleatoric Uncertainty
Estimated via heteroscedastic prediction heads attached to a shared MLP backbone. The model jointly predicts a mean and a scale parameter, trained with the corresponding negative log-likelihood. A conditional Normalizing Flow is optionally applied to the standardized residuals to capture non-Gaussian residual structure.

### Epistemic Uncertainty
Estimated via two approaches:
- **MC Dropout** — T=50 stochastic forward passes at inference time
- **Deep Ensembles** — M=10 independently trained members with different random seeds

### DIDO
A post-hoc auxiliary model that discretizes a residual-based error proxy into bins and outputs Dirichlet concentration parameters. Vacuity (K/total concentration) is used as an epistemic ranking signal.

### Calibration
Global variance scaling parameters (α for aleatoric, β for epistemic) are fitted on a validation split by minimizing Gaussian NLL, then frozen for test set evaluation.

---

## Motivation

In domains like auto finance, insurance, or any regression-based risk system, knowing *when* a model is likely to be wrong is as important as the prediction itself. This project explores whether decomposed uncertainty signals can support:

- **Selective review** — prioritising cases for manual inspection
- **Distribution shift monitoring** — detecting when inputs differ from training data
- **Risk-aware decisions** — acting more conservatively when uncertainty is high

---

## Academic Context

This work was developed as part of a Master's thesis in Data Analytics at Stiftung Universität Hildesheim, under the supervision of Prof. Dr. Dr. Lars Schmidt-Thieme, Ibram Abdelmalak and Jan Schnitker.

---

## License

MIT License — see [LICENSE](LICENSE) for details.
