# Atelier 2 - Mini-projet ML & Git

Baseline reproductible de classification du dataset Iris avec une régression logistique, génération d'artefacts et workflow Git MLOps.

## Installation

```bash
python -m venv .venv
```

Sous Windows PowerShell :

```powershell
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Sous Linux/macOS :

```bash
source .venv/bin/activate
pip install -r requirements.txt
```

## Exécution

```bash
pytest -q
python scripts/train.py
python scripts/evaluate.py
```

## Artefacts générés

- `artifacts/model.joblib`
- `artifacts/metrics.json`
- `artifacts/confusion_matrix.png`
- `artifacts/run_info.json`
- `artifacts/report.json`

## Organisation Git

Le code, la configuration, les tests et la documentation sont versionnés. Les datasets réels, les modèles et les sorties générées restent hors de Git grâce au fichier `.gitignore`.

Workflow recommandé : `main` pour la version stable, `dev` pour l'intégration et `feature/preprocessing` pour l'évolution du prétraitement.

## Partie GitHub à réaliser avec votre compte

1. Créez un dépôt GitHub vide.
2. Ajoutez-le avec `git remote add origin <URL>`.
3. Poussez `main`, `dev` et `feature/preprocessing`.
4. Ouvrez une Pull Request de `feature/preprocessing` vers `dev`.
5. Fusionnez `dev` vers `main`, puis poussez le tag `v0.1.0`.

Le compte rendu fourni contient les preuves locales. Ajoutez vos captures authentiques de la Pull Request et de la release GitHub avant le dépôt final si elles sont exigées.

# Atelier 2 - Mini-projet ML & Git

Le pipeline applique une imputation médiane, une standardisation et un clipping des valeurs entre -3 et 3.