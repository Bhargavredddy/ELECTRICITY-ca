import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.io as pio

sns.set(style='whitegrid', context='notebook')


def plot_pca_scatter(X, labels, output_path):
    fig, ax = plt.subplots(figsize=(9, 6))
    scatter = ax.scatter(X[:, 0], X[:, 1], c=labels, cmap='tab10', alpha=0.8, s=40)
    ax.set_title('PCA Projection of Consumption Patterns')
    ax.set_xlabel('PCA component 1')
    ax.set_ylabel('PCA component 2')
    legend1 = ax.legend(*scatter.legend_elements(), title='Cluster')
    ax.add_artist(legend1)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def plot_hourly_cluster_heatmap(df, labels, output_path):
    hour_cols = [f'h{h}' for h in range(24) if f'h{h}' in df.columns]
    cluster_profiles = (
        df.assign(cluster=labels)
          .groupby('cluster')[hour_cols]
          .mean()
          .sort_index()
    )
    fig, ax = plt.subplots(figsize=(12, 5))
    sns.heatmap(cluster_profiles, cmap='viridis', annot=False, cbar=True, ax=ax)
    ax.set_title('Average Hourly Consumption by Cluster')
    ax.set_xlabel('Hour of day')
    ax.set_ylabel('Cluster label')
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def plot_elbow_curve(X, output_path):
    distortions = []
    k_values = list(range(2, 8))
    from sklearn.cluster import KMeans
    for k in k_values:
        model = KMeans(n_clusters=k, random_state=42, n_init=10)
        model.fit(X)
        distortions.append(model.inertia_)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(k_values, distortions, marker='o')
    ax.set_title('Elbow Curve for K-Means')
    ax.set_xlabel('Number of clusters')
    ax.set_ylabel('Inertia')
    ax.grid(True)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def plot_silhouette_scores(X, output_path):
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score
    scores = []
    k_values = list(range(2, 8))
    for k in k_values:
        model = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = model.fit_predict(X)
        scores.append(silhouette_score(X, labels))
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(k_values, scores, marker='o')
    ax.set_title('Silhouette Scores for K-Means')
    ax.set_xlabel('Number of clusters')
    ax.set_ylabel('Silhouette score')
    ax.grid(True)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def plotly_pca_scatter(X, labels):
    df = pd.DataFrame({
        'pca1': X[:, 0],
        'pca2': X[:, 1],
        'cluster': labels.astype(str),
    })
    fig = px.scatter(
        df,
        x='pca1',
        y='pca2',
        color='cluster',
        labels={'pca1': 'PCA component 1', 'pca2': 'PCA component 2'},
        title='PCA projection of electricity consumption patterns',
        template='plotly_dark',
        hover_data=['cluster'],
        custom_data=['cluster'],
    )
    fig.update_traces(marker={'size': 10, 'opacity': 0.85}, hovertemplate='Cluster: %{customdata[0]}<br>PCA1: %{x}<br>PCA2: %{y}<extra></extra>')
    return pio.to_html(fig, full_html=False, include_plotlyjs='cdn')


def plotly_hourly_cluster_heatmap(df, labels):
    hour_cols = [f'h{h}' for h in range(24) if f'h{h}' in df.columns]
    cluster_profiles = (
        df.assign(cluster=labels)
          .groupby('cluster')[hour_cols]
          .mean()
          .sort_index()
    )
    fig = px.imshow(
        cluster_profiles,
        labels=dict(x='Hour of day', y='Cluster label', color='Average consumption'),
        x=hour_cols,
        y=[str(c) for c in cluster_profiles.index],
        aspect='auto',
        color_continuous_scale='Viridis',
        title='Average hourly consumption by cluster',
        template='plotly_dark',
    )
    fig.update_xaxes(tickmode='array', tickvals=list(range(len(hour_cols))), ticktext=hour_cols)
    return pio.to_html(fig, full_html=False, include_plotlyjs=False)


def plotly_elbow_curve(X):
    distortions = []
    k_values = list(range(2, 8))
    from sklearn.cluster import KMeans
    for k in k_values:
        model = KMeans(n_clusters=k, random_state=42, n_init=10)
        model.fit(X)
        distortions.append(model.inertia_)
    fig = px.line(
        x=k_values,
        y=distortions,
        markers=True,
        labels={'x': 'Number of clusters', 'y': 'Inertia'},
        title='Elbow curve for K-Means',
        template='plotly_dark'
    )
    fig.update_traces(customdata=k_values, hovertemplate='k=%{customdata}<br>Inertia=%{y:.2f}<extra></extra>')
    return pio.to_html(fig, full_html=False, include_plotlyjs=False)


def plotly_silhouette_scores(X):
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score
    scores = []
    k_values = list(range(2, 8))
    for k in k_values:
        model = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = model.fit_predict(X)
        scores.append(silhouette_score(X, labels))
    fig = px.line(
        x=k_values,
        y=scores,
        markers=True,
        labels={'x': 'Number of clusters', 'y': 'Silhouette score'},
        title='Silhouette scores for K-Means',
        template='plotly_dark'
    )
    fig.update_traces(customdata=k_values, hovertemplate='k=%{customdata}<br>Score=%{y:.3f}<extra></extra>')
    return pio.to_html(fig, full_html=False, include_plotlyjs=False)
