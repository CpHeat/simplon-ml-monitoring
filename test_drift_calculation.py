"""
Test manuel du calcul de drift avec Evidently
Pour diagnostiquer pourquoi drift_share ne change pas
"""
import pandas as pd
import sys
from pathlib import Path

# Import des modules
sys.path.insert(0, str(Path(__file__).parent / "API"))
from app.database import get_db_session, TitanicPrediction

print("="*70)
print("DIAGNOSTIC DRIFT CALCULATION")
print("="*70)

# 1. Charger les données de référence
print("\n[1] Chargement données de référence...")
ref_path = "MONITORING/evidently/reference_data/titanic_reference.csv"
reference = pd.read_csv(ref_path)
print(f"  Référence : {reference.shape}")
print(f"  Features: {reference.columns.tolist()}")
print(f"\n  Distribution référence:")
print(f"    Sex: {reference['sex'].value_counts().to_dict()}")
print(f"    Pclass: {reference['pclass'].value_counts().to_dict()}")

# 2. Charger les données de production depuis la DB
print("\n[2] Chargement données production (DB)...")
db = get_db_session()
try:
    predictions = db.query(TitanicPrediction).filter(
        TitanicPrediction.model_type == 'ml'
    ).all()

    data = [{
        'sex': p.sex,
        'pclass': p.pclass,
        'age': p.age,
        'survived': p.prediction,
        'target': p.prediction
    } for p in predictions]

    current = pd.DataFrame(data)
    print(f"  Production : {current.shape}")
    print(f"\n  Distribution production:")
    print(f"    Sex: {current['sex'].value_counts().to_dict()}")
    print(f"    Pclass: {current['pclass'].value_counts().to_dict()}")

finally:
    db.close()

# 3. Calculer le drift avec Evidently
print("\n[3] Calcul du drift avec Evidently...")

from evidently import DataDefinition, Dataset, BinaryClassification, Report
from evidently.presets import DataDriftPreset

# Configuration avec DataDefinition (v0.7+)
# Pour drift detection, on spécifie juste les types de colonnes
data_definition = DataDefinition(
    numerical_columns=['age'],
    categorical_columns=['sex', 'pclass', 'survived']
)

# Créer les Dataset objects
reference_dataset = Dataset.from_pandas(
    reference[['sex', 'pclass', 'age', 'survived']],
    data_definition=data_definition
)
current_dataset = Dataset.from_pandas(
    current[['sex', 'pclass', 'age', 'survived']],
    data_definition=data_definition
)

report = Report(metrics=[
    DataDriftPreset()
])

print("  Exécution du rapport...")
my_eval = report.run(current_dataset, reference_dataset)

# 4. Extraire les métriques
print("\n[4] Résultats du drift:")
import json
report_dict = json.loads(my_eval.json())

drift_share = 0.0
drifted_count = 0
drifted_features = []

for metric in report_dict.get('metrics', []):
    metric_name = metric.get('metric_name', '')

    # Chercher le DriftedColumnsCount
    if 'DriftedColumnsCount' in metric_name:
        value = metric.get('value', {})
        if isinstance(value, dict):
            drift_share = value.get('share', 0.0)
            drifted_count = value.get('count', 0)

    # Collecter les features driftées
    elif 'ValueDrift' in metric_name:
        if 'column=' in metric_name:
            col_start = metric_name.index('column=') + 7
            col_end = metric_name.index(',', col_start)
            column_name = metric_name[col_start:col_end]

            drift_score = metric.get('value', 0)
            if isinstance(drift_score, (int, float)) and drift_score < 0.05:
                drifted_features.append((column_name, drift_score))

print(f"\n  Dataset Drift Detected: {drift_share > 0.5}")
print(f"  Drift Share: {drift_share:.3f}")
print(f"  Number of Drifted Columns: {int(drifted_count)}")

print(f"\n  Drift par feature:")
for col, score in drifted_features:
    print(f"    {col:10} : DRIFT! (p-value={score:.3f})")

print("\n" + "="*70)
print("DIAGNOSTIC TERMINÉ")
print("="*70)
print("\n[INTERPRÉTATION]")
print("Si Drift Share est toujours 0.5 malgré l'injection de données biaisées,")
print("cela signifie que les données de production ne sont pas assez différentes")
print("de la référence, OU que la taille de l'échantillon est trop petite.")
print("\nSolution : Injecter PLUS de données driftées (ex: 100 au lieu de 20)")
print("="*70)
