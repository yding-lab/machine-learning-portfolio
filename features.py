"""Feature engineering and preprocessing.

The module keeps preprocessing inside scikit-learn Pipelines to prevent
train/test leakage and to make saved models directly usable at inference.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler


class ClinicalRatioFeatures(BaseEstimator, TransformerMixin):
    """Create compact domain-inspired ratio features.

    The transformation is deterministic and stateless, making it safe for
    training and production inference.
    """

    def fit(self, X: pd.DataFrame, y=None):
        self.feature_names_in_ = list(X.columns)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        df = X.copy()

        eps = 1e-8
        pairs = {
            "radius_texture_ratio": ("mean radius", "mean texture"),
            "perimeter_area_ratio": ("mean perimeter", "mean area"),
            "worst_radius_texture_ratio": ("worst radius", "worst texture"),
            "compactness_concavity_ratio": (
                "mean compactness",
                "mean concavity",
            ),
        }

        for new_col, (numerator, denominator) in pairs.items():
            if numerator in df.columns and denominator in df.columns:
                df[new_col] = (
                    df[numerator].astype(float)
                    / (df[denominator].astype(float) + eps)
                )

        return df.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    def get_feature_names_out(self, input_features=None):
        base = list(
            input_features
            if input_features is not None
            else getattr(self, "feature_names_in_", [])
        )
        extra = [
            "radius_texture_ratio",
            "perimeter_area_ratio",
            "worst_radius_texture_ratio",
            "compactness_concavity_ratio",
        ]
        return np.asarray(base + extra, dtype=object)


def build_preprocessor(feature_names: list[str]) -> Pipeline:
    """Build the reusable preprocessing pipeline."""
    numeric_transformer = ColumnTransformer(
        transformers=[
            (
                "numeric",
                RobustScaler(),
                feature_names
                + [
                    "radius_texture_ratio",
                    "perimeter_area_ratio",
                    "worst_radius_texture_ratio",
                    "compactness_concavity_ratio",
                ],
            )
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )

    return Pipeline(
        steps=[
            ("ratio_features", ClinicalRatioFeatures()),
            ("scale", numeric_transformer),
        ]
    )
