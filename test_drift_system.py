"""
Script de test pour générer des prédictions et tester le système de drift monitoring.
"""
import requests
import time
import random
from datetime import datetime

API_URL = "http://localhost:8000"

def test_titanic_ml():
    """Test du modèle Titanic ML"""
    print("\n[TEST] Titanic ML predictions...")

    # Générer des prédictions normales
    for i in range(5):
        data = {
            "genre": random.choice(["homme", "femme"]),
            "pclass": random.randint(1, 3),
            "age": random.uniform(1, 80),
            "true_label": random.randint(0, 1)  # Simuler le vrai label
        }

        try:
            response = requests.post(f"{API_URL}/predict/titanic_ml", json=data)
            if response.status_code == 200:
                result = response.json()
                print(f"  ✓ Prediction {i+1}: {result['prediction']} (prob: {result['probability_survive']:.2%})")
            else:
                print(f"  ✗ Error {response.status_code}: {response.text}")
        except Exception as e:
            print(f"  ✗ Exception: {e}")

        time.sleep(0.5)

def test_titanic_dl():
    """Test du modèle Titanic DL"""
    print("\n[TEST] Titanic DL predictions...")

    for i in range(5):
        data = {
            "genre": random.choice(["homme", "femme"]),
            "pclass": random.randint(1, 3),
            "age": random.randint(1, 80),
            "embarked": random.choice(["C", "Q", "S"]),
            "true_label": random.randint(0, 1)
        }

        try:
            response = requests.post(f"{API_URL}/predict/titanic_dl", json=data)
            if response.status_code == 200:
                result = response.json()
                print(f"  ✓ Prediction {i+1}: {result['prediction']} (prob: {result['probability_survive']:.2%})")
            else:
                print(f"  ✗ Error {response.status_code}: {response.text}")
        except Exception as e:
            print(f"  ✗ Exception: {e}")

        time.sleep(0.5)

def test_unemployment():
    """Test du modèle Unemployment"""
    print("\n[TEST] Unemployment predictions...")

    countries = ["France", "USA", "Germany", "Japan", "Brazil"]

    for i in range(5):
        data = {
            "country": random.choice(countries),
            "agriculture": random.uniform(1, 30),
            "industry": random.uniform(10, 40),
            "services": random.uniform(40, 80),
            "gdp_log": random.uniform(24, 30),
            "year": random.randint(2015, 2023)
        }

        try:
            response = requests.post(f"{API_URL}/predict/unemployment", json=data)
            if response.status_code == 200:
                result = response.json()
                print(f"  ✓ {data['country']} ({data['year']}): {result['predicted_unemployment_rate']:.2f}%")
            else:
                print(f"  ✗ Error {response.status_code}: {response.text}")
        except Exception as e:
            print(f"  ✗ Exception: {e}")

        time.sleep(0.5)

def generate_drift_scenario():
    """Générer des données avec drift pour tester la détection"""
    print("\n[DRIFT SCENARIO] Generating predictions with simulated drift...")
    print("  → Toutes les prédictions seront des femmes de 1ère classe (distribution anormale)")

    for i in range(10):
        data = {
            "genre": "femme",  # Seulement des femmes (drift!)
            "pclass": 1,        # Seulement 1ère classe (drift!)
            "age": random.uniform(20, 40),  # Age concentré (drift!)
            "true_label": 1  # Toujours survie
        }

        try:
            response = requests.post(f"{API_URL}/predict/titanic_ml", json=data)
            if response.status_code == 200:
                print(f"  ✓ Drift prediction {i+1} logged")
            else:
                print(f"  ✗ Error {response.status_code}")
        except Exception as e:
            print(f"  ✗ Exception: {e}")

        time.sleep(0.3)

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
    print("="*60)
    print("DRIFT MONITORING SYSTEM - TEST SUITE")
    print("="*60)
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"API URL: {API_URL}")
    print("="*60)

    # Vérifier l'API
    if not check_health():
        print("\n[ERROR] API is not accessible. Please start the FastAPI service first:")
        print("  docker-compose up -d fastapi")
        exit(1)

    # Menu
    print("\n[MENU] Choose test scenario:")
    print("  1. Normal predictions (mixed data)")
    print("  2. Drift scenario (biased data)")
    print("  3. Both (normal + drift)")
    print("  4. Continuous testing (10 rounds)")

    choice = input("\nYour choice [1-4]: ").strip()

    if choice == "1":
        test_titanic_ml()
        test_titanic_dl()
        test_unemployment()

    elif choice == "2":
        generate_drift_scenario()

    elif choice == "3":
        test_titanic_ml()
        test_titanic_dl()
        test_unemployment()
        print("\nWaiting 2 seconds before drift scenario...")
        time.sleep(2)
        generate_drift_scenario()

    elif choice == "4":
        for round_num in range(1, 11):
            print(f"\n{'='*60}")
            print(f"ROUND {round_num}/10")
            print(f"{'='*60}")
            test_titanic_ml()
            test_titanic_dl()
            test_unemployment()

            if round_num < 10:
                print("\nWaiting 5 seconds before next round...")
                time.sleep(5)

    else:
        print("[ERROR] Invalid choice")
        exit(1)

    print("\n" + "="*60)
    print("TEST COMPLETED")
    print("="*60)
    print("\n[NEXT STEPS]")
    print("  1. Check database: MONITORING/evidently/data/predictions.db")
    print("  2. Wait for next drift monitoring run (hourly)")
    print("  3. Check reports: MONITORING/evidently/reports/")
    print("  4. View metrics: http://localhost:8002/metrics")
    print("  5. View Prometheus: http://localhost:9090")
    print("  6. View Grafana: http://localhost:3000")
    print("  7. View Evidently UI: http://localhost:8001")
    print("\n" + "="*60)
