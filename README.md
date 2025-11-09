# Évaluation du risque de crédit

ce dépôt contient un pipeline d'analyse et un modèle pour l'évaluation du risque de crédit.

Contenu principal :
- `data/` : jeux de données (brut et nettoyé).
- `data_preprocessing.py` : préparation des données.
- `credit_risk_model.joblib` : modèle entraîné (chargeable avec joblib).
- `notebooks`: pour l'analyse exploratoire et la modélisation
- `mlruns/` : artefacts et métriques MLflow.

Usage rapide :
1) Créer un environnement Python (>=3.12.5) et installer les dépendances listées dans `pyproject.toml`.
2) Prétraiter les données : `python data_preprocessing.py` (arguments selon le script).
3) Charger le modèle :

```python
import joblib
model = joblib.load('credit_risk_model.joblib')
```

Pour visualiser les expériences : `mlflow ui --backend-store-uri ./mlruns`.



