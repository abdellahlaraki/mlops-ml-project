"""Evalue le modele sauvegarde sur le dataset configure."""

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import joblib
import yaml
from sklearn.metrics import classification_report

from src.data import load_dataset


def main():
    with open("config/train.yaml", "r", encoding="utf-8") as stream:
        cfg = yaml.safe_load(stream)
    output_dir = Path(cfg.get("artifacts_dir", "artifacts"))
    model_path = output_dir / "model.joblib"
    if not model_path.exists():
        raise FileNotFoundError("Lancez d'abord : python scripts/train.py")

    model = joblib.load(model_path)
    X, y = load_dataset(cfg)
    report = classification_report(y, model.predict(X), output_dict=True)
    with open(output_dir / "report.json", "w", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2)
    print("Evaluate OK: artifacts/report.json")


if __name__ == "__main__":
    main()
