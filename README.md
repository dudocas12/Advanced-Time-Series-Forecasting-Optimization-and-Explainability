# Advanced Time Series Forecasting, Optimization and Explainability

**Course:** Advanced Topics in Deep Learning — 2nd Semester 2025/2026

**Dataset:** Jena Climate (Weather)

**Task:** Multivariate-input, univariate-output, multi-step air temperature forecasting

---

## Project Structure

```
├── main.ipynb                  # Main notebook with all experiments
├── data/                       # Data directory (auto-downloaded)
│   └── jena_climate_2009_2016.csv
├── Projecto_final_v1.pdf       # Project assignment specification
├── requirements.txt            # Python dependencies
├── README.md                   # This file
└── .gitignore
```

## Sections

| # | Section | Description |
|---|---------|-------------|
| 1 | Setup & Imports | Libraries, seeds, device config |
| 2 | Data Loading | Download & load Jena Climate dataset |
| 3 | EDA | Exploratory data analysis & visualizations |
| 4 | Preprocessing | Sub-sampling, temporal covariates, normalization, splits |
| 5 | Windowing | Sliding window creation & DataLoaders |
| 6 | Baseline Models | GRU and Transformer baselines |
| 7 | Evolutionary Optimization | EA-based pipeline optimization |
| 8 | Synthetic Data *(bonus)* | GAN-based data augmentation |
| 9 | Explainable AI | Global & local XAI methods |
| 10 | Efficiency Analysis | Training time, inference time, parameters, memory |
| 11 | Discussion | Comparative analysis & trade-offs |
| 12 | Conclusion | Summary & key findings |

## Setup

```bash
pip install -r requirements.txt
```

## Group Members

- Student 1 — Name (ID)
- Student 2 — Name (ID)
- Student 3 — Name (ID)
