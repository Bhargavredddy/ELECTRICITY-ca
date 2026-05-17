import os
import re
from pathlib import Path
from datetime import datetime
from flask import Flask, render_template, request, redirect, url_for, send_file
from werkzeug.utils import secure_filename
import pandas as pd

from src.data_generation import save_synthetic_data
from src.preprocessing import PreprocessingPipeline
from src.clustering import ClusterModels
from src.visualization import (
    plot_pca_scatter,
    plot_hourly_cluster_heatmap,
    plot_elbow_curve,
    plot_silhouette_scores,
    plotly_pca_scatter,
    plotly_hourly_cluster_heatmap,
    plotly_elbow_curve,
    plotly_silhouette_scores,
)

BASE_DIR = Path(__file__).resolve().parents[1]
UPLOAD_FOLDER = BASE_DIR / 'data' / 'uploads'
SAMPLE_DATA_PATH = BASE_DIR / 'data' / 'synthetic_electricity_consumption.csv'
SAMPLE_DOWNLOAD_PATH = BASE_DIR / 'data' / 'sample_upload_format.csv'
STATIC_ANALYSIS_FOLDER = Path(__file__).parent / 'static' / 'analysis'
OUTPUT_FOLDER = BASE_DIR / 'reports'
ALLOWED_EXTENSIONS = {'csv'}

for folder in [UPLOAD_FOLDER, STATIC_ANALYSIS_FOLDER, OUTPUT_FOLDER, SAMPLE_DATA_PATH.parent]:
    folder.mkdir(parents=True, exist_ok=True)


def ensure_sample_dataset():
    if not SAMPLE_DATA_PATH.exists():
        save_synthetic_data(SAMPLE_DATA_PATH, n_customers=400, random_state=42)
    if not SAMPLE_DOWNLOAD_PATH.exists():
        sample = pd.read_csv(SAMPLE_DATA_PATH).head(12)
        sample.to_csv(SAMPLE_DOWNLOAD_PATH, index=False)


def infer_hourly_columns(df: pd.DataFrame):
    exact_cols = [f'h{h}' for h in range(24)]
    if all(col in df.columns for col in exact_cols):
        return exact_cols

    hour_labels = {}
    for col in df.columns:
        normalized = col.strip().lower().replace(' ', '').replace('_', '')
        match = re.match(r'^(?:h|hour|hr)?(0?\d|1\d|2[0-3])$', normalized)
        if match:
            hour_labels[int(match.group(1))] = col
        elif normalized.isdigit():
            value = int(normalized)
            if 0 <= value <= 23:
                hour_labels[value] = col

    if len(hour_labels) == 24:
        return [hour_labels[h] for h in range(24)]

    numeric_cols = [col for col in df.columns if pd.api.types.is_numeric_dtype(df[col])]
    if len(numeric_cols) >= 24:
        return numeric_cols[:24]

    raise ValueError(
        'Unable to infer 24 hourly consumption columns from the uploaded file. '
        'Please upload a dataset with 24 hourly values or use the sample dataset.'
    )


def prepare_consumption_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    hour_cols = infer_hourly_columns(df)
    if hour_cols != [f'h{h}' for h in range(24)]:
        for idx, source_col in enumerate(hour_cols):
            df[f'h{idx}'] = df[source_col]
        hour_cols = [f'h{h}' for h in range(24)]

    df['avg_load'] = df[hour_cols].mean(axis=1)
    df['peak_load'] = df[hour_cols].max(axis=1)
    df['morning_load'] = df[[f'h{h}' for h in range(6, 12)]].mean(axis=1)
    df['evening_load'] = df[[f'h{h}' for h in range(17, 23)]].mean(axis=1)
    df['load_variance'] = df[hour_cols].var(axis=1)
    return df

app = Flask(
    __name__,
    template_folder='templates',
    static_folder='static',
)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024


def allowed_file(filename: str) -> bool:
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def run_analysis(df: pd.DataFrame, original_filename: str):
    df = prepare_consumption_data(df)
    if 'household_id' not in df.columns:
        df['household_id'] = [f'U{i+1}' for i in range(len(df))]

    pipeline = PreprocessingPipeline(n_components=6)
    X_reduced = pipeline.fit_transform(df)
    X_full = pipeline.transform_full(df)

    model = ClusterModels(kmeans_k=4, dbscan_eps=1.5, dbscan_min_samples=12)
    kmeans_labels, dbscan_labels = model.fit(X_full)
    outlier_indices = model.identify_outliers()

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    base_name = secure_filename(Path(original_filename).stem + '_' + timestamp)

    plot_html = {
        'pca': plotly_pca_scatter(X_reduced, kmeans_labels),
        'heatmap': plotly_hourly_cluster_heatmap(df, kmeans_labels),
        'elbow': plotly_elbow_curve(X_reduced),
        'silhouette': plotly_silhouette_scores(X_reduced),
    }

    # keep legacy static assets for offline reporting and dashboard export
    plot_files = {
        'pca': f'{base_name}_pca_clusters.png',
        'heatmap': f'{base_name}_heatmap.png',
        'elbow': f'{base_name}_elbow.png',
        'silhouette': f'{base_name}_silhouette.png',
    }
    plot_pca_scatter(X_reduced, kmeans_labels, STATIC_ANALYSIS_FOLDER / plot_files['pca'])
    plot_hourly_cluster_heatmap(df, kmeans_labels, STATIC_ANALYSIS_FOLDER / plot_files['heatmap'])
    plot_elbow_curve(X_reduced, STATIC_ANALYSIS_FOLDER / plot_files['elbow'])
    plot_silhouette_scores(X_reduced, STATIC_ANALYSIS_FOLDER / plot_files['silhouette'])

    summary = (
        df[['household_id']].copy()
          .assign(kmeans_cluster=kmeans_labels, dbscan_cluster=dbscan_labels)
    )
    summary_path = OUTPUT_FOLDER / f'{base_name}_cluster_summary.csv'
    summary.to_csv(summary_path, index=False)

    cluster_counts = (
        summary.groupby('kmeans_cluster')['household_id']
               .count()
               .reset_index()
               .rename(columns={'household_id': 'count'})
               .sort_values('kmeans_cluster')
               .to_dict(orient='records')
    )

    cluster_feature_means = (
        df.assign(kmeans_cluster=kmeans_labels)
          .groupby('kmeans_cluster')[['avg_load', 'peak_load', 'morning_load', 'evening_load', 'load_variance']]
          .mean()
          .round(3)
          .reset_index()
          .to_dict(orient='records')
    )

    cluster_stats = model.describe_clusters(X_full).round(3).to_dict(orient='records')

    overall_metrics = {
        'avg_load_mean': float(df['avg_load'].mean()),
        'peak_load_mean': float(df['peak_load'].mean()),
        'morning_load_mean': float(df['morning_load'].mean()),
        'evening_load_mean': float(df['evening_load'].mean()),
        'variance_mean': float(df['load_variance'].mean()),
        'hourly_columns': [f'h{h}' for h in range(24)],
    }

    explanations = {
        'preprocessing': (
            'Hourly values are inferred from columns h0 through h23, then normalized and transformed using PCA. '\
            'The pipeline also computes engineered features: avg_load, peak_load, morning_load, evening_load, and load_variance.'
        ),
        'pca': (
            'PCA reduces the 24 hourly features plus engineered metrics into two components so similar consumption patterns appear close in 2D space. '\
            'Each point represents one household day profile.'
        ),
        'heatmap': (
            'The heatmap shows average hourly consumption for each cluster, where rows are clusters and columns are hours of the day. '\
            'Brighter colors represent higher average usage.'
        ),
        'elbow': (
            'The elbow curve plots K-Means inertia for different k values. '\
            'The “elbow” suggests a good number of clusters by showing diminishing gains in compactness.'
        ),
        'silhouette': (
            'Silhouette scores evaluate how well points fit within their assigned clusters compared to other clusters. '\
            'Higher values indicate tighter, more separated clusters.'
        ),
    }

    return {
        'filename': original_filename,
        'count': len(df),
        'outliers': int(len(outlier_indices)),
        'clusters': cluster_counts,
        'cluster_feature_means': cluster_feature_means,
        'cluster_stats': cluster_stats,
        'metrics': overall_metrics,
        'explanations': explanations,
        'plot_files': plot_files,
        'plot_html': plot_html,
        'summary_file': summary_path.name,
    }


@app.route('/', methods=['GET'])
def index():
    return render_template('upload.html')


@app.route('/analyze', methods=['POST'])
def analyze():
    if request.form.get('use_sample'):
        try:
            ensure_sample_dataset()
            df = pd.read_csv(SAMPLE_DATA_PATH)
            analysis = run_analysis(df, SAMPLE_DATA_PATH.name)
            analysis['source'] = 'sample'
            return render_template('results.html', analysis=analysis)
        except Exception as exc:
            return render_template('upload.html', error=f'Sample analysis failed: {str(exc)}')

    if 'csv_file' not in request.files:
        return render_template('upload.html', error='Please choose a CSV file to upload or use the sample data.')

    file = request.files['csv_file']
    if file.filename == '':
        return render_template('upload.html', error='No file selected. Please upload a CSV file or use the sample data.')

    if not allowed_file(file.filename):
        return render_template('upload.html', error='Only CSV files are supported.')

    filename = secure_filename(file.filename)
    upload_path = UPLOAD_FOLDER / filename
    file.save(upload_path)

    try:
        df = pd.read_csv(upload_path)
        analysis = run_analysis(df, filename)
        analysis['source'] = 'upload'
        return render_template('results.html', analysis=analysis)
    except Exception as exc:
        return render_template('upload.html', error=f'Upload failed: {str(exc)}')


@app.route('/download-sample')
def download_sample():
    ensure_sample_dataset()
    return send_file(
        SAMPLE_DOWNLOAD_PATH,
        as_attachment=True,
        download_name='sample_upload_format.csv',
        mimetype='text/csv',
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)