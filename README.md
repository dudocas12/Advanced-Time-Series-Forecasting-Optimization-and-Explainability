# Advanced Time Series Forecasting, Optimization and Explainability

**Course:** Advanced Topics in Deep Learning — 2nd Semester 2025/2026

**Dataset:** [Jena Climate](https://www.bgc-jena.mpg.de/wetter/) (Max Planck Institute for Biogeochemistry)

**Task:** Multivariate-input, univariate-output, multi-step air temperature forecasting

---

## Overview

This project develops, optimizes, and interprets a deep learning pipeline for multi-step weather forecasting. Starting from two baseline architectures (GRU and Transformer), we use an Evolutionary Algorithm to discover an optimal architecture, apply three Explainable AI techniques to interpret model decisions, and evaluate the accuracy–efficiency trade-offs across all models.

---

## Project Structure

```
├── main.ipynb                          # Complete notebook (all experiments + outputs)
├── requirements.txt                    # Python dependencies
├── data/                               # Dataset (auto-downloaded on first run)
│   └── jena_climate_2009_2016.csv
├── models/                             # Pre-trained model weights & EA results
│   ├── gru_baseline.pt
│   ├── transformer_baseline.pt
│   └── species_*/                     # EA evolutionary run checkpoints
├── README.md
└── .gitignore
```

---

## Notebook Sections

|   #   | Section                                 | Description                                                                                         |
| :---: | --------------------------------------- | --------------------------------------------------------------------------------------------------- |
|   1   | **Introduction**                        | Problem motivation, architecture overview, project goals                                            |
|   2   | **Setup & Imports**                     | Libraries, reproducibility seeds, GPU/CPU device configuration                                      |
|   3   | **Dataset Description**                 | Jena Climate dataset loading, variable selection, initial inspection                                |
|   4   | **Exploratory Data Analysis**           | Time series plots, correlation matrix, distributions                                                |
|   5   | **Data Preprocessing**                  | Hourly sub-sampling, cyclic temporal covariates, 70/15/15 train-val-test split, StandardScaler      |
|   6   | **Windowing & DataLoaders**             | Sliding window creation, PyTorch Dataset/DataLoader pipeline                                        |
|   7   | **Baseline Models**                     | GRU and Transformer architectures, training with early stopping, evaluation                         |
|   8   | **Evolutionary Optimization**           | DEAP-based EA search over architecture, hyperparameters, and windowing; multi-seed robustness check |
|   9   | **Synthetic Data Generation** *(bonus)* | Jittering, scaling, and time warping augmentation on training data only                             |
|  10   | **Explainable AI (XAI)**                | Permutation Feature Importance (global), Integrated Gradients & Deep SHAP (local)                   |
|  11   | **Efficiency & Resource Analysis**      | Parameter counts, inference time, GPU memory, accuracy–efficiency comparison table                  |
|  12   | **Discussion**                          | Baseline vs. EA analysis, XAI insights with cross-method validation, trade-off analysis             |
|  13   | **Conclusion**                          | Key findings and future work                                                                        |

---

## Quick Start

```bash
pip install -r requirements.txt
jupyter notebook main.ipynb
```

## Group Members

- David Isaac — 120064
- David Volovei — 120051
- Eduardo Martins — 120063
