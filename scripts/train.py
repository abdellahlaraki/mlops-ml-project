"""Entraine le modele et genere les artefacts MLOps."""

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import sklearn
import yaml
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from src.data import load_dataset
from src.features import build_numeric_preprocess
from src.model import build_model


def load_cfg(path="config/train.yaml"):
    with open(path, "r", encoding="utf-8") as stream:
        return yaml.safe_load(stream)


def save_confusion_matrix(y_true, y_pred, output_path):
    matrix = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    image = ax.imshow(matrix, cmap="Blues")
    fig.colorbar(image, ax=ax)
    ax.set(title="Matrice de confusion", xlabel="Prediction", ylabel="Valeur reelle")
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j, i, str(matrix[i, j]), ha="center", va="center")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def main():
    cfg = load_cfg()
    output_dir = Path(cfg.get("artifacts_dir", "artifacts"))
    output_dir.mkdir(parents=True, exist_ok=True)

    seed = int(cfg["split"]["random_state"])
    np.random.seed(seed)
    X, y = load_dataset(cfg)
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=float(cfg["split"]["test_size"]),
        random_state=seed,
        stratify=y,
    )

    pipeline = Pipeline(
        [("preprocess", build_numeric_preprocess()), ("model", build_model(cfg))]
    )
    pipeline.fit(X_train, y_train)
    predictions = pipeline.predict(X_test)
    metrics = {
        "accuracy": float(accuracy_score(y_test, predictions)),
        "f1_macro": float(f1_score(y_test, predictions, average="macro")),
    }

    joblib.dump(pipeline, output_dir / "model.joblib")
    with open(output_dir / "metrics.json", "w", encoding="utf-8") as stream:
        json.dump(metrics, stream, indent=2)
    save_confusion_matrix(y_test, predictions, output_dir / "confusion_matrix.png")

    run_info = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "configuration": cfg,
        "versions": {"python": sys.version, "numpy": np.__version__, "scikit_learn": sklearn.__version__},
    }
    with open(output_dir / "run_info.json", "w", encoding="utf-8") as stream:
        json.dump(run_info, stream, indent=2)

    print("Train OK")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
