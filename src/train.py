import os
from pathlib import Path
import pandas as pd
from joblib import dump

from src.data_generation import save_synthetic_data, load_consumption_data
from src.preprocessing import PreprocessingPipeline
from src.clustering import ClusterModels
from src.visualization import (
    plot_pca_scatter,
    plot_hourly_cluster_heatmap,
    plot_elbow_curve,
    plot_silhouette_scores,
)


def ensure_dirs(paths):
    for path in paths:
        Path(path).mkdir(parents=True, exist_ok=True)


def build_dataset(data_dir: Path, force_regenerate=False):
    dataset_path = data_dir / 'synthetic_electricity_consumption.csv'
    if dataset_path.exists() and not force_regenerate:
        return load_consumption_data(dataset_path)
    return save_synthetic_data(dataset_path, n_customers=600, random_state=42)


def save_cluster_summary(df, kmeans_labels, dbscan_labels, output_path: Path):
    summary = (
        df[['household_id']].copy()
          .assign(kmeans_cluster=kmeans_labels, dbscan_cluster=dbscan_labels)
    )
    summary.to_csv(output_path, index=False)
    return summary


def main():
    workspace_root = Path(__file__).resolve().parents[1]
    data_dir = workspace_root / 'data'
    output_dir = workspace_root / 'reports'
    models_dir = workspace_root / 'models'
    plots_dir = output_dir / 'plots'

    ensure_dirs([data_dir, output_dir, models_dir, plots_dir])

    print('Generating or loading dataset...')
    df = build_dataset(data_dir)

    print('Running preprocessing pipeline...')
    pipeline = PreprocessingPipeline(n_components=6)
    X_reduced = pipeline.fit_transform(df)
    X_full = pipeline.transform_full(df)

    print('Training clustering models...')
    clusters = ClusterModels(kmeans_k=4, dbscan_eps=1.5, dbscan_min_samples=12)
    kmeans_labels, dbscan_labels = clusters.fit(X_full)
    outlier_indices = clusters.identify_outliers()

    print('Saving cluster models and transforms...')
    dump(pipeline, models_dir / 'preprocessing_pipeline.joblib')
    clusters.save(str(models_dir / 'cluster_models.joblib'))

    print('Saving cluster summary...')
    save_cluster_summary(df, kmeans_labels, dbscan_labels, output_dir / 'cluster_summary.csv')

    print('Writing visualization assets...')
    plot_elbow_curve(X_reduced, plots_dir / 'elbow_curve.png')
    plot_silhouette_scores(X_reduced, plots_dir / 'silhouette_scores.png')
    plot_pca_scatter(X_reduced, kmeans_labels, plots_dir / 'pca_kmeans_clusters.png')
    plot_hourly_cluster_heatmap(df, kmeans_labels, plots_dir / 'hourly_cluster_heatmap.png')

    cluster_description = clusters.describe_clusters(X_full)
    cluster_description.to_csv(output_dir / 'cluster_description.csv', index=False)

    print('Pipeline complete.')
    print(f' - {len(df)} records processed')
    print(f' - {len(outlier_indices)} DBSCAN outliers identified')
    print(f' - Models stored in {models_dir}')
    print(f' - Visualizations stored in {plots_dir}')
    print('Backend is ready for frontend development.')


if __name__ == '__main__':
    main()
