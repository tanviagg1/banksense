# BankSense

Customer segmentation and product recommender built as an end-to-end ML learning project.

Groups bank customers by financial behaviour using unsupervised learning, then recommends the right financial product to each segment.

## What it covers

| Concept | Notebook |
|---|---|
| pandas — DataFrames, filtering, groupby, feature engineering | 01 |
| EDA — distributions, outliers, correlation heatmap | 02 |
| Preprocessing — imputation, scaling, sklearn Pipeline, data leakage | 03 |
| PCA — dimensionality reduction, explained variance, loadings | 04 |
| KMeans — elbow method, silhouette score, Davies-Bouldin | 05 |
| Cluster profiling, segment naming, product recommender | 06 |

## Segments discovered

| Segment | Profile | Recommended Product |
|---|---|---|
| Young Spender | Age ~26, income £32k, savings £2.5k, credit 617 | Premium Cashback Credit Card |
| Stable Saver | Age ~42, income £59k, savings £35k, credit 732 | High-Yield Savings Account |
| High Earner | Age ~46, income £118k, savings £89k, credit 788 | Wealth Management Portfolio |
| Debt Stressed | Age ~37, income £28k, savings £1.2k, loan £34k | Debt Consolidation Loan |

KMeans recovered all 4 segments with 97-100% purity — without ever seeing the labels.

## Setup

```bash
git clone https://github.com/tanviagg1/banksense.git
cd banksense
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/jupyter notebook
```

Open notebooks in order: `01 → 06`. Each notebook is self-contained with explanations of what and why before every code block.

## Project structure

```
banksense/
├── notebooks/
│   ├── 01_data_generation.ipynb
│   ├── 02_eda.ipynb
│   ├── 03_preprocessing.ipynb
│   ├── 04_pca.ipynb
│   ├── 05_clustering.ipynb
│   └── 06_analysis.ipynb
├── src/
│   ├── data_generator.py     # generates 1000 synthetic banking customers
│   └── recommender.py        # predict_segment_and_recommend() for new customers
├── data/                     # auto-created when you run the notebooks
└── requirements.txt
```

## Stack

- Python 3.10+
- pandas, numpy, scikit-learn, matplotlib, seaborn, jupyter
