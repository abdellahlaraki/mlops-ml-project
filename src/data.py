"""Chargement des donnees Iris ou d'un fichier CSV."""

import pandas as pd
from sklearn.datasets import load_iris


def load_dataset(cfg: dict):
    data_cfg = cfg["data"]
    kind = data_cfg.get("kind", "iris").lower()

    if kind == "iris":
        return load_iris(return_X_y=True, as_frame=True)

    if kind == "csv":
        path = data_cfg["path"]
        target = data_cfg["target"]
        df = pd.read_csv(path)
        if target not in df.columns:
            raise ValueError(f"Colonne cible absente : {target}")
        return df.drop(columns=[target]), df[target]

    raise ValueError(f"Type de donnees non pris en charge : {kind}")
