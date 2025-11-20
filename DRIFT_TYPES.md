# Types de Drift - Guide Complet

Ce document explique les trois types de drift détectés par le système de monitoring.

---

## 📊 1. DATA DRIFT (Drift des Features)

### Définition
Le **data drift** se produit quand la distribution des **variables d'entrée** (features) change par rapport aux données d'entraînement.

### Exemples concrets (Titanic)

**Distribution de référence** (dataset d'entraînement) :
- 65% hommes, 35% femmes
- 55% classe 3, 21% classe 2, 24% classe 1
- Âge moyen : ~30 ans (écart-type: 14)

**Data drift détecté** si en production :
- 90% hommes, 10% femmes → Drift sur `sex`
- 80% classe 1, 10% classe 2, 10% classe 3 → Drift sur `pclass`
- Âge moyen : 50 ans → Drift sur `age`

### Comment c'est détecté ?

**Tests statistiques utilisés** :
- **Variables numériques** (`age`, `gdp_log`) : Test de Kolmogorov-Smirnov
  - Compare les distributions continues
  - p-value < 0.05 → Drift détecté
- **Variables catégorielles** (`sex`, `pclass`, `embarked`) : Test du Chi²
  - Compare les proportions des catégories
  - p-value < 0.05 → Drift détecté

### Métriques Prometheus

```promql
# Drift global (% de features ayant drifté)
ml_drift_share{model="titanic_ml"}

# Drift par feature
ml_feature_drift_detected{model="titanic_ml", feature="age"}
ml_feature_drift_score{model="titanic_ml", feature="age"}
```

### Panels Grafana
- **Bar Gauge** : "Data Drift by Feature" (Panel 14-16)
  - Montre quelles features spécifiques driftent
- **Time Series** : "Feature Drift Scores Over Time" (Panel 17)
  - Évolution du score de drift (p-value) dans le temps

### Causes possibles
- Changement de la population utilisatrice
- Évolution des conditions métier
- Saisonnalité
- Erreur dans la collecte de données

### Actions recommandées
1. ✅ **Analyser le rapport HTML** `titanic_ml_drift_YYYYMMDD_HHMMSS.html`
2. ✅ **Identifier les features problématiques**
3. ✅ **Décider** :
   - Réentraîner le modèle avec nouvelles données ?
   - Adapter le preprocessing ?
   - Investiguer la cause du changement ?

---

## 🎯 2. TARGET DRIFT (Drift de la Variable Cible)

### Définition
Le **target drift** se produit quand la distribution de la **variable cible** change, indépendamment des features.

### Exemples concrets (Titanic)

**Distribution de référence** :
- 38% survie, 62% décès

**Target drift détecté** si en production :
- 70% survie, 30% décès → La proportion de survie a drastiquement changé
- Ou inversement : 10% survie, 90% décès

### Différence avec Data Drift

| Type | Change quoi ? | Exemple |
|------|---------------|---------|
| **Data Drift** | Distribution des **inputs** | Âge moyen passe de 30 à 50 ans |
| **Target Drift** | Distribution de l'**output** | Taux de survie passe de 38% à 70% |

Les deux peuvent se produire **indépendamment** :
- Data drift SANS target drift : Inputs changent, mais taux de survie reste 38%
- Target drift SANS data drift : Inputs identiques, mais taux de survie monte à 70%

### Comment c'est détecté ?

**Méthode** :
- Comparaison de la distribution de `survived` (0 ou 1)
- Test du Chi² sur les proportions
- p-value < 0.05 → Target drift détecté

**Prérequis** :
- Nécessite des `true_label` en production
- Sans true_labels, impossible de détecter le target drift

### Métriques Prometheus

```promql
# Target drift détecté ?
ml_target_drift_detected{model="titanic_ml"}
```

### Panels Grafana
- **Stat Panel** : "Target Drift Status" (Panel 18)
  - Indicateur rouge/vert du statut actuel
- **Time Series** : "Target Drift Over Time" (Panel 19)
  - Historique des événements de target drift

### Causes possibles
- Changement du comportement métier (ex: nouvelles procédures de sauvetage)
- Biais dans l'échantillonnage des données de production
- Évolution naturelle du phénomène

### Actions recommandées
1. ✅ **Vérifier si c'est un vrai changement** ou un biais d'échantillonnage
2. ✅ **Investiguer la cause métier**
3. ✅ **Réentraîner le modèle** si le changement est permanent
4. ✅ **Ajuster les seuils de décision** si seule la proportion change

---

## 🔄 3. CONCEPT DRIFT (Drift du Concept)

### Définition
Le **concept drift** se produit quand la **relation entre les features et la target** change. Les inputs peuvent rester identiques, mais leur signification change.

### Exemples concrets (Titanic)

**Concept de référence** :
- Femme + Classe 1 + Jeune → 95% de chances de survie
- Homme + Classe 3 + Âgé → 5% de chances de survie

**Concept drift détecté** si :
- Femme + Classe 1 + Jeune → **50% de survie** (au lieu de 95%)
  - La relation a changé : être une femme de classe 1 ne garantit plus la survie
- Les **mêmes inputs** donnent maintenant des **outputs différents**

### Différence avec les autres drifts

| Type | Change quoi ? | Impact |
|------|---------------|--------|
| **Data Drift** | Distribution des features | Inputs différents, mais relations identiques |
| **Target Drift** | Distribution de la target | Proportion globale change |
| **Concept Drift** | **Relation features → target** | Les règles du jeu changent |

**Exemple combiné** :
```
Data Drift    : Âge moyen passe de 30 à 50 ans
Target Drift  : Taux de survie reste 38%
Concept Drift : Avant, "jeune = +20% survie", maintenant "jeune = 0% impact"
                → La relation âge → survie a changé
```

### Comment c'est détecté ?

**Méthode indirecte** (notre implémentation) :
- Concept drift = **dégradation des performances du modèle**
- Si F1 Score descend, c'est que la relation a changé
- Le modèle prédit mal car les règles qu'il a apprises ne sont plus valides

**Indicateurs** :
- F1 Score < 0.7 → ⚠️ Warning
- F1 Score < 0.5 → 🚨 Critique
- Accuracy, Precision, Recall en baisse simultanée

**Méthode directe** (nécessite Evidently avancé) :
- Comparer les corrélations features ↔ target
- Détecter les changements de relations statistiques

### Métriques Prometheus

```promql
# Performance du modèle (indicateur de concept drift)
ml_model_f1_score{model="titanic_ml"}
ml_model_accuracy{model="titanic_ml"}
ml_model_precision{model="titanic_ml"}
ml_model_recall{model="titanic_ml"}

# Corrélation performance vs drift
ml_model_f1_score / ml_drift_share
```

### Panels Grafana
- **Time Series** : "Concept Drift Detection" (Panel 20)
  - F1 Score avec seuils à 0.7 et 0.5
  - Dégradation = concept drift probable
- **Dual-Axis** : "Performance vs Drift Correlation" (Panel 21)
  - F1 Score (gauche) vs Drift Share (droite)
  - Permet de voir si le drift cause la dégradation

### Causes possibles
- Évolution du contexte métier (nouvelles règles)
- Changement de comportement des utilisateurs
- Événement externe impactant les relations
- Dataset d'entraînement non représentatif de la réalité actuelle

### Actions recommandées
1. 🚨 **Réentraîner le modèle immédiatement** avec nouvelles données
2. ✅ **Investiguer pourquoi** la relation a changé (analyse métier)
3. ✅ **Vérifier les features** : sont-elles toujours pertinentes ?
4. ✅ **Considérer de nouvelles features** capturant mieux le nouveau concept
5. ✅ **Implémenter un système de réentraînement automatique** si drift fréquent

---

## 📊 Tableau comparatif

| Aspect | Data Drift | Target Drift | Concept Drift |
|--------|------------|--------------|---------------|
| **Quoi ?** | Distribution des features | Distribution de la target | Relation features → target |
| **Exemple Titanic** | 90% hommes (vs 65%) | 70% survie (vs 38%) | Femme classe 1 → 50% survie (vs 95%) |
| **Test statistique** | K-S, Chi² sur features | Chi² sur target | Dégradation F1 Score |
| **Détectable sans labels ?** | ✅ Oui | ❌ Non | ❌ Non |
| **Métrique Prometheus** | `ml_feature_drift_detected` | `ml_target_drift_detected` | `ml_model_f1_score` |
| **Panel Grafana** | Panel 14-17 | Panel 18-19 | Panel 20-21 |
| **Gravité** | ⚠️ Moyenne | ⚠️ Moyenne | 🚨 Haute |
| **Action** | Investiguer | Vérifier | Réentraîner immédiatement |

---

## 🔍 Workflow de diagnostic

Quand une alerte se déclenche, suivez ce workflow :

### 1️⃣ Identifier le type de drift

```bash
# Consulter Prometheus
curl http://localhost:8002/metrics | grep drift

# Ou Grafana → Dashboard Enhanced
```

**Questions** :
- ✅ `ml_dataset_drift_detected` = 1 ? → **Data Drift**
- ✅ `ml_target_drift_detected` = 1 ? → **Target Drift**
- ✅ `ml_model_f1_score` < 0.7 ? → **Concept Drift**

### 2️⃣ Analyser les rapports HTML

```bash
# Ouvrir le dernier rapport
cd MONITORING/evidently/reports
ls -lt titanic_ml_drift_*.html | head -1
```

**Data Drift Report** :
- Quelles features driftent ?
- Quelle est l'ampleur du drift ?
- Graphiques de distribution

**Performance Report** :
- Matrice de confusion
- Accuracy, Precision, Recall, F1
- Nombre de true_labels disponibles

### 3️⃣ Corréler avec les panels Grafana

**Dashboard Enhanced** :
- Row 2 : Data Drift par feature
- Row 3 : Target Drift
- Row 5 : Concept Drift (Performance vs Drift)

### 4️⃣ Décider de l'action

| Scénario | Action |
|----------|--------|
| **Data Drift seul** (F1 score OK) | Surveiller, investiguer la cause |
| **Target Drift seul** | Vérifier l'échantillonnage, analyser métier |
| **Concept Drift** (F1 < 0.7) | 🚨 **Réentraîner immédiatement** |
| **Data + Concept Drift** | Réentraîner avec nouvelles données |
| **Tous les drifts** | Crise majeure, réentraînement urgent + investigation approfondie |

---

## 🎯 Métriques par type de drift

### Data Drift
```promql
ml_drift_share{model="titanic_ml"}                    # Proportion de features driftées
ml_feature_drift_detected{feature="age"}              # Drift sur feature spécifique
ml_feature_drift_score{model="titanic_ml"}            # p-value du test
```

### Target Drift
```promql
ml_target_drift_detected{model="titanic_ml"}          # 0 ou 1
```

### Concept Drift
```promql
ml_model_f1_score{model="titanic_ml"}                 # Indicateur principal
ml_model_accuracy{model="titanic_ml"}                 # Confirmation
ml_model_f1_score < 0.7                               # Alerte warning
ml_model_f1_score < 0.5                               # Alerte critique
```

---

## 📚 Ressources

- [PROMETHEUS_METRICS.md](PROMETHEUS_METRICS.md) : Liste complète des métriques
- [REPORTS_STRUCTURE.md](REPORTS_STRUCTURE.md) : Structure des rapports HTML
- [Evidently Documentation](https://docs.evidentlyai.com/user-guide/tests-and-reports/drift-detection)
- Dashboard Grafana : `ml-monitoring-enhanced.json`

---

## 💡 Bonnes pratiques

1. ✅ **Monitorer les 3 types** de drift systématiquement
2. ✅ **Prioriser le concept drift** (impact immédiat sur le business)
3. ✅ **Automatiser les alertes** Prometheus pour F1 < 0.7
4. ✅ **Maintenir des true_labels** en production pour détecter target/concept drift
5. ✅ **Réentraîner régulièrement** même sans drift (tous les 3-6 mois)
6. ✅ **Documenter les causes** de drift pour apprentissage futur
