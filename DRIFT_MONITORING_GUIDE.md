# 📊 Guide du Système de Monitoring de Drift ML avec Evidently AI

## 🎯 Vue d'ensemble

Ce système implémente un monitoring complet du drift ML avec 3 types de drift :
- **Data Drift** : Changement dans la distribution des features
- **Target Drift** : Changement dans la distribution des prédictions
- **Concept Drift** : Dégradation de la performance (Titanic uniquement)

### Architecture

```
┌─────────────┐     ┌──────────────┐     ┌───────────────┐
│   FastAPI   │────▶│  SQLite DB   │────▶│ Drift Monitor │
│   (8000)    │     │ predictions  │     │    (8002)     │
└─────────────┘     └──────────────┘     └───────┬───────┘
                                                 │
                                                 ▼
┌─────────────┐     ┌──────────────┐     ┌───────────────┐
│  Evidently  │◀────│  Prometheus  │◀────│   Métriques   │
│  UI (8001)  │     │    (9090)    │     │   de Drift    │
└─────────────┘     └──────┬───────┘     └───────────────┘
                           │
                           ▼
                    ┌──────────────┐
                    │   Grafana    │
                    │   (3000)     │
                    └──────────────┘
```

## 🚀 Démarrage du système

### 1. Prérequis

```bash
# Installer les dépendances Python
pip install -r API/requirements.txt

# Ou si vous utilisez uniquement Docker
docker-compose build
```

### 2. Générer les datasets de référence

```bash
cd MONITORING/evidently/reference_data
python extract_reference_data.py
```

Résultat attendu :
```
[OK] Titanic reference saved: (712, 6)
[OK] Unemployment reference saved: (4600, 8)
```

### 3. Démarrer tous les services

```bash
# Démarrer tous les conteneurs
docker-compose up -d

# Vérifier que tous les services sont up
docker-compose ps
```

Vous devriez voir **6 conteneurs** :
- ✅ `fastapi` (8000)
- ✅ `prometheus` (9090)
- ✅ `grafana` (3000)
- ✅ `cadvisor` (8080)
- ✅ `evidently-ui` (8001)
- ✅ `drift-monitor` (8002)

### 4. Vérifier les services

| Service | URL | Description |
|---------|-----|-------------|
| API FastAPI | http://localhost:8000/docs | Documentation Swagger |
| Prometheus | http://localhost:9090 | Métriques brutes |
| Grafana | http://localhost:3000 | Dashboards (admin/admin) |
| Evidently UI | http://localhost:8001 | Rapports de drift |
| Drift Metrics | http://localhost:8002/metrics | Métriques Prometheus |

## 📈 Configuration du Dashboard Grafana (Étape par étape)

### Étape 1 : Se connecter à Grafana

1. Aller sur http://localhost:3000
2. Login : `admin` / `admin` (ou les credentials dans `.env`)
3. Changer le mot de passe si demandé

### Étape 2 : Vérifier la datasource Prometheus

1. Menu (☰) → **Connections** → **Data sources**
2. Vérifier que `Prometheus` existe
3. Si pas présente :
   - Cliquer **Add data source**
   - Choisir **Prometheus**
   - URL : `http://prometheus:9090`
   - Cliquer **Save & test**

### Étape 3 : Créer le Dashboard de Drift

1. Menu (☰) → **Dashboards** → **New** → **New Dashboard**
2. Cliquer **Add visualization**
3. Sélectionner datasource **Prometheus**

#### Panel 1 : Drift Share (Line Chart)

**Configuration :**
- **Panel title** : "Drift Share par Modèle"
- **Query** :
  ```promql
  ml_drift_share
  ```
- **Legend** : `{{model}}`
- **Unit** : Percent (0.0-1.0)
- **Graph type** : Time series

#### Panel 2 : Drift Détecté (Gauge)

**Configuration :**
- **Panel title** : "Drift Détecté (Statut Actuel)"
- **Query** :
  ```promql
  ml_dataset_drift_detected
  ```
- **Visualization** : Gauge
- **Thresholds** :
  - 0 = Vert (No Drift)
  - 1 = Rouge (Drift Detected)
- **Value mappings** :
  - 0 → "No Drift"
  - 1 → "Drift Detected"

#### Panel 3 : Nombre de Features Driftées

**Configuration :**
- **Panel title** : "Nombre de Features Driftées"
- **Query** :
  ```promql
  ml_drifted_columns_count
  ```
- **Legend** : `{{model}}`
- **Graph type** : Time series

#### Panel 4 : Volume de Prédictions

**Configuration :**
- **Panel title** : "Prédictions (dernière période)"
- **Query** :
  ```promql
  ml_predictions_count_last_period
  ```
- **Legend** : `{{model}}`
- **Graph type** : Time series

#### Panel 5 : Accuracy des Modèles (Titanic)

**Configuration :**
- **Panel title** : "Accuracy Titanic (Concept Drift)"
- **Query** :
  ```promql
  ml_model_accuracy{model=~"titanic.*"}
  ```
- **Legend** : `{{model}}`
- **Unit** : Percent (0.0-1.0)
- **Thresholds** :
  - > 0.7 = Vert
  - 0.6-0.7 = Jaune
  - < 0.6 = Rouge

#### Panel 6 : Échantillons avec Labels

**Configuration :**
- **Panel title** : "Échantillons avec Labels"
- **Query** :
  ```promql
  ml_samples_with_labels
  ```
- **Visualization** : Stat
- **Legend** : `{{model}}`

### Étape 4 : Sauvegarder le Dashboard

1. Cliquer sur l'icône **💾 Save** en haut
2. Nom : "ML Drift Monitoring"
3. Cliquer **Save**

### Étape 5 : Configurer le Refresh

1. En haut à droite, cliquer sur l'icône **⟳**
2. Sélectionner **30s** ou **1m**

## 🧪 Tester le Système

### Test 1 : Prédictions Normales

```bash
python test_drift_system.py
```

Choisir l'option **1** : Normal predictions

### Test 2 : Simuler du Drift

Choisir l'option **2** : Drift scenario

Cela va générer des prédictions avec une distribution biaisée (seulement femmes, 1ère classe) pour déclencher la détection de drift.

### Test 3 : Test Continu

Choisir l'option **4** : Continuous testing

Génère 10 rounds de prédictions pour accumuler des données.

## 📊 Visualiser les Résultats

### 1. Vérifier la Database

```bash
# Installer sqlite3 si besoin
pip install sqlite3

# Ouvrir la DB
sqlite3 MONITORING/evidently/data/predictions.db

# Compter les prédictions
SELECT model_type, COUNT(*) FROM titanic_predictions GROUP BY model_type;
SELECT COUNT(*) FROM unemployment_predictions;
```

### 2. Consulter les Métriques Prometheus

Aller sur http://localhost:8002/metrics

Métriques clés :
```
ml_dataset_drift_detected{model="titanic_ml"}
ml_drift_share{model="titanic_ml"}
ml_drifted_columns_count{model="titanic_ml"}
ml_model_accuracy{model="titanic_ml"}
ml_predictions_count_last_period{model="titanic_ml"}
```

### 3. Voir les Rapports Evidently

#### Option A : Rapports HTML

```bash
# Les rapports sont générés dans
cd MONITORING/evidently/reports

# Ouvrir le dernier rapport
start titanic_ml_drift_YYYYMMDD_HHMMSS.html
```

#### Option B : Evidently UI

1. Aller sur http://localhost:8001
2. Uploader les rapports depuis `MONITORING/evidently/reports/`

### 4. Dashboard Grafana

Aller sur http://localhost:3000 et consulter votre dashboard créé.

## ⚙️ Configuration Avancée

### Changer la Fréquence de Monitoring

Éditer `MONITORING/evidently/service/scheduler.py` :

```python
# Ligne 203 : Changer interval_hours
scheduler = DriftScheduler(interval_hours=1)  # 1h par défaut
```

Puis rebuild :
```bash
docker-compose up -d --build drift-monitor
```

### Ajuster les Seuils d'Alertes

Éditer `MONITORING/prometheus/alerts.yml` :

```yaml
# Exemple : Alerte drift à 20% au lieu de 30%
- alert: HighDriftShare
  expr: ml_drift_share > 0.2  # Changé de 0.3 à 0.2
```

Redémarrer Prometheus :
```bash
docker-compose restart prometheus
```

## 🔍 Monitoring Manuel

Si vous voulez lancer le monitoring manuellement (sans attendre 1h) :

```bash
# Entrer dans le conteneur
docker exec -it drift-monitor python drift_monitor.py
```

Ou en local :
```bash
cd MONITORING/evidently/service
python drift_monitor.py
```

## 📋 Métriques Disponibles

| Métrique | Description | Type | Labels |
|----------|-------------|------|--------|
| `ml_dataset_drift_detected` | Drift détecté (0 ou 1) | Gauge | model |
| `ml_drift_share` | Pourcentage de features driftées | Gauge | model |
| `ml_drifted_columns_count` | Nombre de colonnes driftées | Gauge | model |
| `ml_model_accuracy` | Accuracy du modèle | Gauge | model |
| `ml_samples_with_labels` | Samples avec vrai label | Gauge | model |
| `ml_predictions_count_last_period` | Nb prédictions dernière période | Gauge | model |
| `ml_monitoring_runs_total` | Total runs monitoring | Counter | model, status |
| `ml_last_monitoring_run_timestamp` | Timestamp dernier run | Gauge | model |

## 🚨 Alertes Configurées

| Alerte | Condition | Sévérité | Description |
|--------|-----------|----------|-------------|
| `DataDriftDetected` | drift détecté | Warning | Au moins une feature a drifté |
| `HighDriftShare` | > 30% features driftées | Warning | Drift modéré |
| `CriticalDriftShare` | > 50% features driftées | Critical | Drift critique |
| `ModelAccuracyDrop` | Accuracy < 70% | Warning | Performance en baisse |
| `ModelAccuracyCritical` | Accuracy < 60% | Critical | Performance critique |
| `NoPredictionsReceived` | 0 prédiction en 6h | Info | Modèle inactif |
| `DriftMonitoringDown` | Service down | Critical | Infrastructure |

## 🐛 Troubleshooting

### Le drift-monitor ne démarre pas

```bash
# Vérifier les logs
docker logs drift-monitor

# Problème commun : DB path
# Vérifier que le volume est bien monté
docker inspect drift-monitor | grep Mounts -A 20
```

### Pas de métriques dans Prometheus

1. Vérifier que drift-monitor répond :
   ```bash
   curl http://localhost:8002/metrics
   ```

2. Vérifier la config Prometheus :
   ```bash
   docker exec prometheus cat /etc/prometheus/prometheus.yml
   ```

3. Vérifier les targets dans Prometheus UI :
   http://localhost:9090/targets

### Les rapports ne se génèrent pas

```bash
# Vérifier qu'il y a des prédictions dans la DB
docker exec drift-monitor python -c "
from drift_monitor import DriftMonitor
m = DriftMonitor()
data, count = m.get_recent_titanic_predictions(hours=24)
print(f'Predictions found: {count}')
"
```

## 📚 Ressources

- [Documentation Evidently](https://docs.evidentlyai.com/)
- [Prometheus Alerting](https://prometheus.io/docs/alerting/latest/overview/)
- [Grafana Dashboards](https://grafana.com/docs/grafana/latest/dashboards/)

## ✅ Checklist de Validation

- [ ] Les 6 conteneurs Docker sont up
- [ ] Database SQLite créée dans `MONITORING/evidently/data/`
- [ ] Datasets de référence dans `MONITORING/evidently/reference_data/`
- [ ] Script de test génère des prédictions sans erreur
- [ ] Métriques visibles sur http://localhost:8002/metrics
- [ ] Prometheus scrape drift-monitor (check targets)
- [ ] Dashboard Grafana créé avec 6 panels
- [ ] Rapports HTML générés dans `MONITORING/evidently/reports/`
- [ ] Alertes configurées dans Prometheus

---

**Fait avec ❤️ pour le projet académique de monitoring ML**
