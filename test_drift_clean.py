"""
Script de test pour valider le système de drift monitoring.
Utilise des vraies données du dataset Titanic pour avoir des true_labels connus.
"""
import requests
import time
import random
import numpy as np
import pandas as pd
from datetime import datetime

API_URL = "http://localhost:8000"

# Charger le vrai dataset Titanic au démarrage
try:
    TITANIC_DATA = pd.read_csv('ML/data/titanic.csv')
    # Nettoyer les données manquantes
    TITANIC_DATA = TITANIC_DATA.dropna(subset=['Age', 'Sex', 'Pclass', 'Survived', 'Embarked'])
    print(f"[INFO] Loaded Titanic dataset: {len(TITANIC_DATA)} valid rows")
except Exception as e:
    print(f"[WARNING] Could not load Titanic dataset: {e}")
    TITANIC_DATA = None

def find_closest_passenger(sex, pclass, age):
    """
    Trouve les passagers les plus proches dans le dataset réel.
    Si plusieurs passagers ont le même âge (exact ou le plus proche),
    calcule la moyenne de leur survie pour déterminer le true_label.
    Retourne 1 si moyenne >= 0.5, sinon 0.
    """
    if TITANIC_DATA is None:
        # Fallback : estimation simple si dataset non chargé
        survive_prob = 0.38
        if sex == "femme":
            survive_prob += 0.4
        if pclass == 1:
            survive_prob += 0.2
        elif pclass == 3:
            survive_prob -= 0.2
        return 1 if random.random() < survive_prob else 0

    # Convertir le sexe au format du dataset
    sex_orig = 'female' if sex == 'femme' else 'male'

    # Filtrer par sexe et classe (critères catégoriels exacts)
    candidates = TITANIC_DATA[
        (TITANIC_DATA['Sex'] == sex_orig) &
        (TITANIC_DATA['Pclass'] == pclass)
    ]

    if len(candidates) == 0:
        # Si aucun candidat, relâcher la contrainte de classe
        candidates = TITANIC_DATA[TITANIC_DATA['Sex'] == sex_orig]

    if len(candidates) == 0:
        # Dernier recours : tout le dataset
        candidates = TITANIC_DATA

    # Trouver l'âge le plus proche
    candidates = candidates.copy()  # Éviter SettingWithCopyWarning
    candidates['age_diff'] = abs(candidates['Age'] - age)
    min_age_diff = candidates['age_diff'].min()

    # Sélectionner TOUS les passagers ayant cet âge le plus proche
    closest_passengers = candidates[candidates['age_diff'] == min_age_diff]

    # Calculer la moyenne de survie
    survival_mean = closest_passengers['Survived'].mean()

    # Retourner 1 si moyenne >= 0.5, sinon 0
    return 1 if survival_mean >= 0.5 else 0

def generate_normal_titanic_ml(n=20):
    """
    Génère des prédictions Titanic ML qui SUIVENT la distribution de référence.
    Inputs : générés selon les distributions réelles
    True labels : extraits du passager le plus proche dans le dataset
    """
    print(f"\n[NORMAL DATA] Generating {n} Titanic ML predictions (no drift expected)...")
    print("  Distribution basée sur le dataset de référence")

    for i in range(n):
        # Générer des inputs selon les distributions réelles
        # sex: 65% homme, 35% femme
        sex = random.choices(["homme", "femme"], weights=[65, 35])[0]

        # pclass: 55% classe 3, 21% classe 2, 24% classe 1
        pclass = random.choices([1, 2, 3], weights=[24, 21, 55])[0]

        # age: Distribution normale, moyenne ~30 ans, écart-type ~14
        age = max(0, min(80, np.random.normal(30, 14)))

        # Trouver le true_label du passager le plus proche dans le dataset
        true_label = find_closest_passenger(sex, pclass, age)

        data = {
            "genre": sex,
            "pclass": pclass,
            "age": float(age),
            "true_label": true_label
        }

        try:
            response = requests.post(f"{API_URL}/predict/titanic_ml", json=data)
            if response.status_code == 200:
                result = response.json()
                status = "✓" if result['prediction'] == true_label else "✗"
                print(f"  {status} {i+1}/{n}: {sex[:1].upper()}, class {pclass}, age {age:.0f} → pred={result['prediction']}, true={true_label}")
            else:
                print(f"  ✗ Error {response.status_code}")
        except Exception as e:
            print(f"  ✗ Exception: {e}")

        time.sleep(0.3)

    print(f"  → {n} prédictions normales générées")

def generate_normal_titanic_dl(n=15):
    """
    Génère des prédictions Titanic DL qui SUIVENT la distribution de référence.
    Inputs : générés selon les distributions réelles
    True labels : extraits du passager le plus proche dans le dataset
    """
    print(f"\n[NORMAL DATA] Generating {n} Titanic DL predictions (no drift expected)...")

    for i in range(n):
        # Générer des inputs selon les distributions réelles
        sex = random.choices(["homme", "femme"], weights=[65, 35])[0]
        pclass = random.choices([1, 2, 3], weights=[24, 21, 55])[0]
        age = int(max(0, min(80, np.random.normal(30, 14))))

        # embarked: 72% S, 19% C, 9% Q (distribution réelle Titanic)
        embarked = random.choices(["S", "C", "Q"], weights=[72, 19, 9])[0]

        # Trouver le true_label du passager le plus proche
        true_label = find_closest_passenger(sex, pclass, age)

        data = {
            "genre": sex,
            "pclass": pclass,
            "age": age,
            "embarked": embarked,
            "true_label": true_label
        }

        try:
            response = requests.post(f"{API_URL}/predict/titanic_dl", json=data)
            if response.status_code == 200:
                result = response.json()
                status = "✓" if result['prediction'] == true_label else "✗"
                print(f"  {status} {i+1}/{n}: {sex[:1].upper()}, class {pclass}, age {age}, port {embarked} → pred={result['prediction']}, true={true_label}")
            else:
                print(f"  ✗ Error {response.status_code}")
        except Exception as e:
            print(f"  ✗ Exception: {e}")

        time.sleep(0.3)

    print(f"  → {n} prédictions normales générées")

def generate_normal_unemployment(n=15):
    """Génère des prédictions Unemployment normales (distribution mondiale)"""
    print(f"\n[NORMAL DATA] Generating {n} Unemployment predictions (no drift expected)...")

    # Charger la distribution de référence pour la reproduire exactement
    try:
        import pandas as pd
        ref = pd.read_csv('MONITORING/evidently/reference_data/unemployment_reference.csv')
        all_countries = ref['country'].unique().tolist()
        year_range = (int(ref['year'].min()), int(ref['year'].max()))
        print(f"  → Référence : {year_range[0]}-{year_range[1]}, {len(all_countries)} pays")
    except:
        # Fallback si impossible de lire la référence
        all_countries = [
            "France", "Germany", "USA", "Japan", "Brazil", "India", "China",
            "Spain", "Italy", "Canada", "Australia", "Mexico", "South Africa",
            "Russia", "UK", "Indonesia", "Turkey", "Poland", "Belgium", "Sweden"
        ]
        year_range = (1991, 2022)

    for i in range(n):
        country = random.choice(all_countries)

        # Distribution réaliste reproduisant la référence (moyenne ~28%, std ~24%)
        agriculture = max(0.1, min(92, np.random.normal(28.7, 23.9)))
        industry = random.uniform(10, 35)
        services = 100 - agriculture - industry  # Complément

        gdp_log = random.uniform(24, 30)  # PIB varié
        year = random.randint(year_range[0], year_range[1])  # Distribution sur toute la période

        data = {
            "country": country,
            "agriculture": agriculture,
            "industry": industry,
            "services": services,
            "gdp_log": gdp_log,
            "year": year
        }

        try:
            response = requests.post(f"{API_URL}/predict/unemployment", json=data)
            if response.status_code == 200:
                result = response.json()
                print(f"  ✓ {i+1}/{n}: {country} ({year}) → {result['predicted_unemployment_rate']:.2f}%")
            else:
                print(f"  ✗ Error {response.status_code}")
        except Exception as e:
            print(f"  ✗ Exception: {e}")

        time.sleep(0.3)

    print(f"  → {n} prédictions normales générées")

def generate_drifted_titanic_ml(n=20):
    """
    Génère des prédictions avec DRIFT VOLONTAIRE.
    → Seulement des femmes de 1ère classe, jeunes
    Inputs : générés avec distribution biaisée (drift)
    True labels : extraits du passager le plus proche dans le dataset
    """
    print(f"\n[DRIFT DATA] Generating {n} Titanic ML predictions (DRIFT expected)...")
    print("  🚨 Distribution anormale : 100% femmes, 100% classe 1, âge < 30")

    for i in range(n):
        # Générer des inputs avec DRIFT volontaire
        sex = "femme"  # 100% femmes (vs 35% référence) → DRIFT!
        pclass = 1      # 100% classe 1 (vs 24% référence) → DRIFT!
        age = float(random.uniform(18, 30))  # Jeunes seulement → DRIFT!

        # Trouver le true_label du passager le plus proche
        true_label = find_closest_passenger(sex, pclass, age)

        data = {
            "genre": sex,
            "pclass": pclass,
            "age": age,
            "true_label": true_label
        }

        try:
            response = requests.post(f"{API_URL}/predict/titanic_ml", json=data)
            if response.status_code == 200:
                result = response.json()
                status = "✓" if result['prediction'] == true_label else "✗"
                print(f"  {status} 🔴 {i+1}/{n}: F, class {pclass}, age {age:.0f} → pred={result['prediction']}, true={true_label}")
            else:
                print(f"  ✗ Error {response.status_code}")
        except Exception as e:
            print(f"  ✗ Exception: {e}")

        time.sleep(0.3)

    print(f"  → {n} prédictions DRIFTÉES générées")

def generate_drifted_unemployment(n=15):
    """Génère des prédictions Unemployment avec drift (seulement France, années récentes)"""
    print(f"\n[DRIFT DATA] Generating {n} Unemployment predictions (DRIFT expected)...")
    print("  🚨 Distribution anormale : 100% France, 100% 2023")

    for i in range(n):
        data = {
            "country": "France",  # 100% France → DRIFT!
            "agriculture": random.uniform(1.5, 3),  # Valeurs France
            "industry": random.uniform(15, 20),
            "services": random.uniform(77, 82),
            "gdp_log": random.uniform(27, 28),
            "year": 2023  # Seulement 2023 → DRIFT!
        }

        try:
            response = requests.post(f"{API_URL}/predict/unemployment", json=data)
            if response.status_code == 200:
                result = response.json()
                print(f"  🔴 {i+1}/{n}: France (2023) → {result['predicted_unemployment_rate']:.2f}%")
            else:
                print(f"  ✗ Error {response.status_code}")
        except Exception as e:
            print(f"  ✗ Exception: {e}")

        time.sleep(0.3)

    print(f"  → {n} prédictions DRIFTÉES générées")

def check_health():
    """Vérifier que l'API est accessible"""
    try:
        response = requests.get(f"{API_URL}/health")
        if response.status_code == 200:
            print("[✓] API is healthy")
            return True
        else:
            print(f"[✗] API returned status {response.status_code}")
            return False
    except Exception as e:
        print(f"[✗] Cannot reach API: {e}")
        return False

if __name__ == "__main__":
    print("="*70)
    print("DRIFT MONITORING - CLEAN TEST SCENARIO")
    print("="*70)
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"API URL: {API_URL}")
    print("="*70)

    if not check_health():
        print("\n[ERROR] API is not accessible.")
        exit(1)

    print("\n[SCENARIO] Test complet de détection de drift")
    print("\nÉtape 1 : Génération de données NORMALES (pas de drift attendu)")
    print("Étape 2 : Attendre 15 minutes pour le monitoring")
    print("Étape 3 : Génération de données DRIFTÉES")
    print("Étape 4 : Observer l'évolution dans Grafana")
    print("\n" + "="*70)

    print("\n[MENU]")
    print("  1. PHASE 1 : Générer des données NORMALES (pas de drift)")
    print("  2. PHASE 2 : Générer des données DRIFTÉES (drift volontaire)")
    print("  3. FULL TEST : Phase 1 + attente + Phase 2 (automatique)")

    choice = input("\nVotre choix [1-3]: ").strip()

    if choice == "1":
        print("\n" + "="*70)
        print("PHASE 1 : DONNÉES NORMALES")
        print("="*70)
        generate_normal_titanic_ml(20)
        generate_normal_titanic_dl(15)
        generate_normal_unemployment(15)

        print("\n" + "="*70)
        print("PHASE 1 TERMINÉE")
        print("="*70)
        print("\n[NEXT STEPS]")
        print("  1. Attendre 15 minutes (prochain run du scheduler)")
        print("  2. Vérifier Grafana : http://localhost:3000")
        print("     → ml_drift_share devrait être FAIBLE (< 0.3)")
        print("     → ml_dataset_drift_detected devrait être 0 (no drift)")
        print("  3. Puis lancer 'python test_drift_clean.py' option 2")

    elif choice == "2":
        print("\n" + "="*70)
        print("PHASE 2 : DONNÉES DRIFTÉES")
        print("="*70)
        generate_drifted_titanic_ml(20)
        generate_drifted_unemployment(15)

        print("\n" + "="*70)
        print("PHASE 2 TERMINÉE")
        print("="*70)
        print("\n[NEXT STEPS]")
        print("  1. Attendre 15 minutes (prochain run du scheduler)")
        print("  2. Vérifier Grafana : http://localhost:3000")
        print("     → ml_drift_share devrait être ÉLEVÉ (> 0.5)")
        print("     → ml_dataset_drift_detected devrait être 1 (drift!)")
        print("     → Alertes devraient se déclencher dans Prometheus")

    elif choice == "3":
        print("\n" + "="*70)
        print("FULL AUTO TEST")
        print("="*70)

        # Phase 1
        print("\n[PHASE 1] Génération données NORMALES...")
        generate_normal_titanic_ml(20)
        generate_normal_titanic_dl(15)
        generate_normal_unemployment(15)

        print("\n[WAITING] Attente de 16 minutes pour le prochain run du scheduler...")
        print("  (Vous pouvez suivre les logs : docker-compose logs -f drift-monitor)")

        for minutes_left in range(16, 0, -1):
            print(f"  ⏳ {minutes_left} minutes restantes...", end='\r')
            time.sleep(60)

        print("\n")

        # Phase 2
        print("\n[PHASE 2] Génération données DRIFTÉES...")
        generate_drifted_titanic_ml(20)
        generate_drifted_unemployment(15)

        print("\n" + "="*70)
        print("TEST COMPLET TERMINÉ")
        print("="*70)
        print("\nAttendre encore 15 minutes puis vérifier Grafana pour voir l'évolution!")

    else:
        print("[ERROR] Choix invalide")
        exit(1)

    print("\n" + "="*70)
