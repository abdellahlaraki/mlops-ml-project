import yaml

from src.data import load_dataset
from src.features import build_numeric_preprocess
from src.model import build_model


def test_dataset_and_components():
    with open("config/train.yaml", "r", encoding="utf-8") as stream:
        cfg = yaml.safe_load(stream)
    X, y = load_dataset(cfg)
    assert X.shape == (150, 4)
    assert len(y) == 150
    assert build_numeric_preprocess() is not None
    assert build_model(cfg).max_iter == 2000
