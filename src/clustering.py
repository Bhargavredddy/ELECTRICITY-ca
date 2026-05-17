import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score
from joblib import dump, load


class ClusterModels:
    def __init__(self, kmeans_k=4, dbscan_eps=1.5, dbscan_min_samples=12):
        self.kmeans_k = kmeans_k
        self.dbscan_eps = dbscan_eps
        self.dbscan_min_samples = dbscan_min_samples
        self.kmeans = KMeans(n_clusters=kmeans_k, random_state=42, n_init=10)
        self.dbscan = DBSCAN(eps=dbscan_eps, min_samples=dbscan_min_samples)
        self.kmeans_labels_ = None
        self.dbscan_labels_ = None
        self.X_train_ = None

    def fit(self, X: np.ndarray):
        self.X_train_ = X
        self.kmeans_labels_ = self.kmeans.fit_predict(X)
        self.dbscan_labels_ = self.dbscan.fit_predict(X)
        return self.kmeans_labels_, self.dbscan_labels_

    def silhouette_score(self, X: np.ndarray, labels: np.ndarray):
        valid = labels != -1
        if valid.sum() < 2:
            return float('nan')
        return silhouette_score(X[valid], labels[valid])

    def describe_clusters(self, X: np.ndarray):
        descriptions = []
        for label in np.unique(self.kmeans_labels_):
            mask = self.kmeans_labels_ == label
            descriptions.append({
                'cluster_label': int(label),
                'count': int(mask.sum()),
                'mean_distance': float(np.linalg.norm(X[mask] - self.kmeans.cluster_centers_[label], axis=1).mean()),
                'max_distance': float(np.linalg.norm(X[mask] - self.kmeans.cluster_centers_[label], axis=1).max()),
            })
        return pd.DataFrame(descriptions)

    def identify_outliers(self):
        return np.where(self.dbscan_labels_ == -1)[0]

    def assign_new_data(self, X_new: np.ndarray):
        if self.kmeans_labels_ is None:
            raise RuntimeError('ClusterModels must be fitted before new data can be assigned.')
        labels = self.kmeans.predict(X_new)
        center_distances = np.linalg.norm(X_new - self.kmeans.cluster_centers_[labels], axis=1)
        threshold = np.percentile(
            np.linalg.norm(self.X_train_ - self.kmeans.cluster_centers_[self.kmeans_labels_], axis=1),
            90,
        )
        labels[center_distances > threshold] = -1
        return labels, center_distances

    def save(self, path: str):
        dump(self, path)

    @staticmethod
    def load(path: str):
        return load(path)
