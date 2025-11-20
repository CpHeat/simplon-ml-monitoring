# Métriques Prometheus - ML Monitoring

Ce document liste toutes les métriques exposées par le système de monitoring de drift.

## 📊 Endpoint des métriques

**URL**: `http://localhost:8002/metrics`

## 🎯 Métriques disponibles

### 1. Drift Detection Metrics

#### `ml_dataset_drift_detected{model="<model_name>"}`
- **Type**: Gauge
- **Description**: Indique si un drift a été détecté sur le dataset
- **Valeurs**:
  - `1` = Drift détecté (plus de 50% des features ont drifté)
  - `0` = Pas de drift
- **Labels**: `model` (titanic_ml, titanic_dl, unemployment)

#### `ml_drift_share{model="<model_name>"}`
- **Type**: Gauge
- **Description**: Proportion de features ayant drifté
- **Valeurs**: `0.0` à `1.0`
- **Exemple**: `0.33` signifie que 33% des features ont drifté
- **Labels**: `model`

#### `ml_drifted_columns_count{model="<model_name>"}`
- **Type**: Gauge
- **Description**: Nombre de colonnes/features ayant drifté
- **Valeurs**: Entier ≥ 0
- **Labels**: `model`

#### `ml_feature_drift_detected{model="<model_name>", feature="<feature_name>"}`
- **Type**: Gauge
- **Description**: Drift détecté pour une feature spécifique
- **Valeurs**:
  - `1` = Drift détecté sur cette feature
  - `0` = Pas de drift
- **Labels**:
  - `model` (titanic_ml, titanic_dl, unemployment)
  - `feature` (sex, pclass, age, embarked, agriculture, industry, etc.)
- **Exemple**: `ml_feature_drift_detected{model="titanic_ml", feature="age"}` = 1

#### `ml_feature_drift_score{model="<model_name>", feature="<feature_name>"}`
- **Type**: Gauge
- **Description**: Score de drift pour une feature (p-value du test statistique)
- **Valeurs**: `0.0` à `1.0` (p-value, plus faible = plus de drift)
- **Interprétation**: p-value < 0.05 indique un drift statistiquement significatif
- **Labels**: `model`, `feature`

#### `ml_target_drift_detected{model="<model_name>"}`
- **Type**: Gauge
- **Description**: Drift détecté sur la variable cible (survived pour Titanic)
- **Valeurs**:
  - `1` = Target drift détecté
  - `0` = Pas de target drift
- **Labels**: `model`
- **Utilité**: Détecte si la distribution de la variable cible a changé

---

### 2. Performance Metrics (Classification - Titanic seulement)

#### `ml_model_accuracy{model="<model_name>"}`
- **Type**: Gauge
- **Description**: Accuracy du modèle basée sur les true_labels
- **Formule**: `(TP + TN) / (TP + TN + FP + FN)`
- **Valeurs**: `0.0` à `1.0`
- **Labels**: `model`

#### `ml_model_precision{model="<model_name>"}`
- **Type**: Gauge
- **Description**: Précision du modèle
- **Formule**: `TP / (TP + FP)`
- **Valeurs**: `0.0` à `1.0`
- **Interprétation**: Parmi les prédictions positives, combien sont correctes
- **Labels**: `model`

#### `ml_model_recall{model="<model_name>"}`
- **Type**: Gauge
- **Description**: Rappel (sensibilité) du modèle
- **Formule**: `TP / (TP + FN)`
- **Valeurs**: `0.0` à `1.0`
- **Interprétation**: Parmi tous les cas positifs réels, combien sont détectés
- **Labels**: `model`

#### `ml_model_f1_score{model="<model_name>"}`
- **Type**: Gauge
- **Description**: F1 Score (moyenne harmonique de precision et recall)
- **Formule**: `2 × (Precision × Recall) / (Precision + Recall)`
- **Valeurs**: `0.0` à `1.0`
- **Interprétation**: Équilibre entre precision et recall
- **Labels**: `model`

---

### 3. Confusion Matrix Metrics

#### `ml_model_true_positives{model="<model_name>"}`
- **Type**: Gauge
- **Description**: Nombre de vrais positifs (prédits survie et vraiment survécu)
- **Valeurs**: Entier ≥ 0
- **Labels**: `model`

#### `ml_model_true_negatives{model="<model_name>"}`
- **Type**: Gauge
- **Description**: Nombre de vrais négatifs (prédits mort et vraiment mort)
- **Valeurs**: Entier ≥ 0
- **Labels**: `model`

#### `ml_model_false_positives{model="<model_name>"}`
- **Type**: Gauge
- **Description**: Nombre de faux positifs (prédits survie mais mort)
- **Valeurs**: Entier ≥ 0
- **Labels**: `model`

#### `ml_model_false_negatives{model="<model_name>"}`
- **Type**: Gauge
- **Description**: Nombre de faux négatifs (prédits mort mais survécu)
- **Valeurs**: Entier ≥ 0
- **Labels**: `model`

---

### 4. Data Volume Metrics

#### `ml_predictions_count_last_period{model="<model_name>"}`
- **Type**: Gauge
- **Description**: Nombre de prédictions dans la dernière période analysée
- **Valeurs**: Entier ≥ 0
- **Labels**: `model`

#### `ml_samples_with_labels{model="<model_name>"}`
- **Type**: Gauge
- **Description**: Nombre d'échantillons avec des true_labels disponibles
- **Valeurs**: Entier ≥ 0
- **Labels**: `model`

---

### 5. Monitoring Service Metrics

#### `ml_monitoring_runs_total{model="<model_name>", status="<status>"}`
- **Type**: Counter
- **Description**: Nombre total d'exécutions du monitoring
- **Labels**:
  - `model`
  - `status` (success, error)

#### `ml_last_monitoring_run_timestamp{model="<model_name>"}`
- **Type**: Gauge
- **Description**: Timestamp Unix de la dernière exécution
- **Valeurs**: Unix timestamp
- **Labels**: `model`

#### `ml_monitoring_info`
- **Type**: Info
- **Description**: Informations sur le service de monitoring
- **Labels statiques**:
  - `version`: Version du service
  - `frequency`: Fréquence d'exécution
  - `models`: Liste des modèles monitorés

---

## 📈 Exemples de requêtes PromQL

### Drift Detection

```promql
# Modèles avec drift détecté
ml_dataset_drift_detected == 1

# Drift share pour tous les modèles
ml_drift_share

# Alerter si drift share > 50%
ml_drift_share > 0.5

# Features spécifiques avec drift (Titanic ML)
ml_feature_drift_detected{model="titanic_ml", feature="age"} == 1

# Toutes les features qui driftent (tous modèles)
ml_feature_drift_detected == 1

# Score de drift pour la feature "age" sur tous les modèles
ml_feature_drift_score{feature="age"}

# Target drift détecté
ml_target_drift_detected == 1

# Nombre de features driftées par modèle
count(ml_feature_drift_detected == 1) by (model)
```

### Performance Monitoring

```promql
# Accuracy par modèle
ml_model_accuracy{model="titanic_ml"}

# F1 Score en dessous de 0.7
ml_model_f1_score < 0.7

# Évolution de la precision sur 1h
rate(ml_model_precision{model="titanic_ml"}[1h])
```

### Confusion Matrix Analysis

```promql
# Taux de faux positifs
ml_model_false_positives / (ml_model_false_positives + ml_model_true_negatives)

# Taux de faux négatifs
ml_model_false_negatives / (ml_model_false_negatives + ml_model_true_positives)
```

### Service Health

```promql
# Dernière exécution (en secondes)
time() - ml_last_monitoring_run_timestamp

# Taux de succès des runs
rate(ml_monitoring_runs_total{status="success"}[5m]) /
rate(ml_monitoring_runs_total[5m])
```

---

## 🚨 Alertes recommandées

Voir le fichier `MONITORING/prometheus/alerts.yml` pour les règles d'alerting configurées.

### Alertes critiques suggérées :

1. **Drift détecté**: `ml_dataset_drift_detected == 1` pendant > 15min
2. **Performance dégradée**: `ml_model_f1_score < 0.6` pendant > 30min
3. **Pas de données**: `ml_predictions_count_last_period == 0` pendant > 1h
4. **Service down**: `time() - ml_last_monitoring_run_timestamp > 3600` (pas de run depuis 1h)

---

## 🔄 Fréquence de mise à jour

- **Intervalle par défaut**: 15 minutes (configurable via `INTERVAL_HOURS` dans scheduler.py)
- **Fenêtre d'analyse**: 24 heures (configurable via paramètre `hours`)

---

## 📊 Dashboard Grafana

Les dashboards Grafana pré-configurés utilisent ces métriques et sont disponibles dans `MONITORING/grafana/dashboards/`.

Pour importer :
1. Aller sur http://localhost:3000
2. Configuration → Data Sources → Ajouter Prometheus (http://prometheus:9090)
3. Dashboards → Import → Charger les fichiers JSON
