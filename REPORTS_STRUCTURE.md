# Structure des Rapports Evidently

Ce document décrit les différents types de rapports générés par le système de monitoring.

## 📁 Localisation des rapports

**Répertoire**: `MONITORING/evidently/reports/`

Les rapports sont générés automatiquement par le scheduler toutes les 15 minutes.

---

## 📊 Types de rapports

Pour chaque modèle Titanic (ML et DL), **4 rapports distincts** sont générés à chaque exécution :

### 1. 📈 Data Drift Report

**Nom du fichier**: `titanic_{ml|dl}_data_drift_YYYYMMDD_HHMMSS.html`

**Contenu**:
- Drift des **features d'entrée** uniquement
- Table détaillée du drift par feature
- Statistiques de distribution pour chaque feature
- Tests statistiques utilisés (Kolmogorov-Smirnov pour numériques, Chi² pour catégoriels)

**Métriques extraites**:
- `drift_share`: Proportion de features ayant drifté
- `dataset_drift_detected`: Booléen (true si > 50% des features driftent)
- `number_of_drifted_columns`: Nombre de features driftées
- `drifted_features`: Liste des features ayant drifté

**Features analysées**:
- **Titanic ML**: `sex`, `pclass`, `age`
- **Titanic DL**: `sex`, `pclass`, `age`, `embarked`

**Utilité**:
- Détecter si les **inputs** changent par rapport à la distribution d'entraînement
- Identifier quelles features spécifiques causent le drift

---

### 2. 🎯 Target Drift Report

**Nom du fichier**: `titanic_{ml|dl}_target_drift_YYYYMMDD_HHMMSS.html`

**Contenu**:
- Drift de la **variable cible** (`survived`)
- Distribution des classes (proportion survive/death)
- Équilibre des classes (class balance)
- Comparaison avec la distribution de référence

**Métriques**:
- Distribution actuelle vs référence
- Tests de drift sur la target

**Utilité**:
- Détecter si la **proportion de survie** change
- Identifier un déséquilibre dans les prédictions

---

### 3. 🔄 Concept Drift Report

**Nom du fichier**: `titanic_{ml|dl}_concept_drift_YYYYMMDD_HHMMSS.html`

**Contenu**:
- Analyse de la **relation entre features et target**
- Métriques de qualité de classification (basées sur prédictions vs true_labels)
- Matrice de confusion
- Distribution des probabilités de prédiction

**Métriques Evidently**:
- Classification Quality Metric
- Confusion Matrix
- Probability Distribution

**Prérequis**:
- ⚠️ Nécessite **au moins 10 true_labels** dans les données de production
- Si pas assez de true_labels, ce rapport est **skippé**

**Utilité**:
- Détecter si la **relation entre inputs et outputs** a changé
- Même si les features restent similaires, le concept peut avoir drifté
- Par exemple : avant, "femme + classe 1" = forte probabilité de survie, maintenant cette relation a changé

---

### 4. ⚡ Performance Report

**Nom du fichier**: `titanic_{ml|dl}_performance_YYYYMMDD_HHMMSS.html`

**Contenu**:
- Rapport HTML personnalisé avec design moderne
- **4 métriques principales** en cartes colorées :
  - ✅ **Accuracy**: Proportion de prédictions correctes
  - 🎯 **Precision**: Qualité des prédictions positives
  - 📊 **Recall**: Capacité à détecter les cas positifs
  - 🔥 **F1 Score**: Équilibre entre precision et recall
- **Matrice de confusion** détaillée avec TP, TN, FP, FN

**Métriques calculées**:
```python
{
    'accuracy': 0.82,
    'precision': 0.85,
    'recall': 0.78,
    'f1_score': 0.81,
    'true_positives': 15,
    'true_negatives': 27,
    'false_positives': 3,
    'false_negatives': 5,
    'samples_with_labels': 50
}
```

**Prérequis**:
- ⚠️ Nécessite **true_labels** dans les données de production
- Si pas de true_labels, ce rapport est **skippé**

**Utilité**:
- Mesurer la **performance réelle** du modèle en production
- Identifier les types d'erreurs (FP vs FN)
- Suivre l'évolution de la performance dans le temps

---

## 📊 Modèle Unemployment

**Nom du fichier**: `unemployment_drift_YYYYMMDD_HHMMSS.html`

Pour le modèle de régression Unemployment, **un seul rapport** est généré :
- Data Drift (features uniquement)
- Pas de target drift (car régression, pas classification)
- Pas de performance metrics (pas de true_labels disponibles)

**Features analysées**:
- `agriculture`, `industry`, `services`, `gdp_log`, `year` (numériques)
- `country` (catégoriel)

---

## 🔄 Cycle de vie des rapports

### Génération
- **Fréquence**: Toutes les 15 minutes (configurable)
- **Fenêtre d'analyse**: 24 heures (données des dernières 24h)
- **Timestamp**: Format `YYYYMMDD_HHMMSS` (ex: `20250120_143022`)

### Stockage
- Les rapports **s'accumulent** dans le répertoire
- Pas de suppression automatique (à nettoyer manuellement si besoin)
- Taille typique : 500 KB - 2 MB par rapport HTML

### Accès
- Ouvrir directement les fichiers HTML dans un navigateur
- Rapports interactifs avec graphiques Plotly
- Navigation entre sections via le menu latéral

---

## 📈 Utilisation des rapports

### Pour les Data Scientists

1. **Data Drift Report** → Comprendre pourquoi les prédictions changent
2. **Target Drift Report** → Vérifier si le problème business a évolué
3. **Concept Drift Report** → Identifier si le modèle doit être réentraîné
4. **Performance Report** → Mesurer l'impact business

### Pour les DevOps/MLOps

1. **Prometheus metrics** → Alerting automatique
2. **Grafana dashboards** → Monitoring temps réel
3. **Rapports HTML** → Analyse post-mortem après alerte

### Workflow typique

```
1. Alerte Prometheus: "Drift détecté sur titanic_ml"
   ↓
2. Consulter Grafana: Drift share = 0.67 (67% des features)
   ↓
3. Ouvrir Data Drift Report: Identifier que "age" et "pclass" ont drifté
   ↓
4. Ouvrir Performance Report: F1 score descendu à 0.65 (vs 0.82 avant)
   ↓
5. Décision: Réentraîner le modèle avec les nouvelles données
```

---

## 🎨 Personnalisation

### Modifier le design du Performance Report

Éditer la fonction `_save_performance_html()` dans `drift_monitor.py` :
- CSS dans la balise `<style>`
- Structure HTML dans le template
- Couleurs des cartes via classes `.green`, `.blue`, `.orange`, `.red`

### Ajouter de nouveaux rapports

Dans `generate_titanic_drift_report()`, ajouter un nouveau bloc :

```python
# 5. NOUVEAU RAPPORT
print(f"  [5/5] Generating Custom report...")
custom_report = Report(metrics=[
    # Vos métriques Evidently ici
])
custom_eval = custom_report.run(current_dataset, reference_dataset)
custom_path = REPORTS_DIR / f"titanic_{model_type}_custom_{timestamp}.html"
custom_eval.save_html(str(custom_path))
report_paths['custom'] = str(custom_path)
```

---

## 📚 Ressources

- [Documentation Evidently](https://docs.evidentlyai.com/)
- [Metrics disponibles](https://docs.evidentlyai.com/reference/all-metrics)
- [Tests de drift statistiques](https://docs.evidentlyai.com/user-guide/tests-and-reports/drift-detection)

---

## ⚠️ Notes importantes

1. **True labels requis** : Les rapports Concept Drift et Performance nécessitent des true_labels
2. **Volume de données** : Minimum 30 échantillons recommandés pour des tests statistiques fiables
3. **Seuil de drift** : Par défaut, drift détecté si p-value < 0.05
4. **Stockage** : Prévoir ~10 MB/jour avec la fréquence actuelle (4 rapports × 3 modèles × 96 runs/jour)
