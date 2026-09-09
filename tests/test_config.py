from pathlib import Path

import yaml


def test_training_configuration():
    with open("config/train.yaml", "r", encoding="utf-8") as stream:
        cfg = yaml.safe_load(stream)
    assert cfg["data"]["kind"] in {"iris", "csv"}
    assert 0 < float(cfg["split"]["test_size"]) < 1
    assert int(cfg["split"]["random_state"]) >= 0
    assert cfg["model"]["name"] == "logistic_regression"
    assert Path(cfg["artifacts_dir"]).name == "artifacts"
