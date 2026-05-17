import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


class PreprocessingPipeline:
    def __init__(self, n_components=6):
        self.imputer = SimpleImputer(strategy='median')
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=n_components, random_state=42)
        self.feature_columns = []
        self.fitted = False

    def _feature_columns(self, df: pd.DataFrame):
        basic_columns = [c for c in df.columns if c.startswith('h') and c[1:].isdigit()]
        engineered_columns = ['avg_load', 'peak_load', 'morning_load', 'evening_load', 'load_variance']
        return [c for c in basic_columns + engineered_columns if c in df.columns]

    def fit_transform(self, df: pd.DataFrame):
        self.feature_columns = self._feature_columns(df)
        X = df[self.feature_columns].copy()
        X = self.imputer.fit_transform(X)
        X = self.scaler.fit_transform(X)
        X_reduced = self.pca.fit_transform(X)
        self.fitted = True
        return X_reduced

    def transform(self, df: pd.DataFrame):
        if not self.fitted:
            raise RuntimeError('PreprocessingPipeline must be fitted before transform.')
        X = df[self.feature_columns].copy()
        X = self.imputer.transform(X)
        X = self.scaler.transform(X)
        return self.pca.transform(X)

    def transform_full(self, df: pd.DataFrame):
        if not self.fitted:
            raise RuntimeError('PreprocessingPipeline must be fitted before transform.')
        X = df[self.feature_columns].copy()
        X = self.imputer.transform(X)
        return self.scaler.transform(X)
