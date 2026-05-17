# ELECTRICITY-ca

## Project: Electricity Consumption Pattern Discovery for Smart Grid Load Balancing

This repository contains the backend implementation for an unsupervised learning project focused on electricity consumption clustering.

Features:
- Synthetic electricity load generation with noise and missing-value fault tolerance
- K-Means and DBSCAN clustering on consumption feature space
- Pattern and region analysis, including dense cluster identification, sparse regions, and outlier discovery
- Visualization outputs: elbow plot, silhouette scores, PCA cluster scatter, and hourly cluster heatmap
- Saved preprocessing pipeline and clustering models for frontend integration

## Setup

Install dependencies:

```bash
pip install -r requirements.txt
```

## Run the backend pipeline

```bash
python -m src.train
```

Outputs are saved to:
- `data/synthetic_electricity_consumption.csv`
- `reports/plots/`
- `models/preprocessing_pipeline.joblib`
- `models/cluster_models.joblib`
- `reports/cluster_summary.csv`
- `reports/cluster_description.csv`

## Frontend readiness

The backend pipeline is ready for frontend development. It produces clustered electricity consumption data, model artifacts, and visualization assets that can be used for dashboard construction, segmentation reporting, or load balancing decision support.

## Web frontend

A dark-themed Flask frontend is now included with two dedicated pages:
- `/` — upload a CSV file or use the sample dataset
- analysis results page — shows cluster statistics, outlier counts, and interactive graphs for PCA, heatmap, elbow, and silhouette score

The upload page supports a wide range of CSV formats by inferring 24 hourly values from columns named like `h0..h23`, `hour0..hour23`, `0..23`, or other numeric hour labels.

Run the frontend web app with:

```bash
python3 -m src.app
```

Then open `http://127.0.0.1:5000` in a browser.
