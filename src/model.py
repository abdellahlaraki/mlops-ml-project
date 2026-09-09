"""Construction du modele baseline."""

from sklearn.linear_model import LogisticRegression


def build_model(cfg: dict):
    model_cfg = cfg["model"]
    name = model_cfg.get("name", "logistic_regression")
    if name != "logistic_regression":
        raise ValueError(f"Modele non pris en charge : {name}")
    return LogisticRegression(
        max_iter=int(model_cfg.get("max_iter", 2000)),
        random_state=int(cfg["split"].get("random_state", 42)),
    )
