"""Pipeline de preparation des variables numeriques."""

import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler


def _clip_values(values):
    """Limite les valeurs standardisees entre -3 et 3."""
    return np.clip(values, -3, 3)


def build_numeric_preprocess():
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("clip", FunctionTransformer(_clip_values)),
        ]
    )
