"""
Script pour extraire les datasets de référence depuis les données d'entraînement.
Ces datasets serviront de baseline pour la détection de drift avec Evidently.
"""
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split

# Chemins
BASE_DIR = Path(__file__).parent.parent.parent.parent
ML_DATA_DIR = BASE_DIR / "ML" / "data"
OUTPUT_DIR = Path(__file__).parent

print(f"[INFO] Recherche des donnees dans: {ML_DATA_DIR}")
print(f"[INFO] Sauvegarde dans: {OUTPUT_DIR}")

# ===== 1. TITANIC REFERENCE DATA =====
print("\n[TITANIC] Extraction donnees Titanic...")

titanic_file = ML_DATA_DIR / "titanic.csv"

if titanic_file.exists():
    df_titanic = pd.read_csv(titanic_file)

    print(f"   Dataset brut: {df_titanic.shape}")
    print(f"   Colonnes: {df_titanic.columns.tolist()}")

    # Preprocessing similaire au modèle
    # Sélection des colonnes utilisées par les modèles
    df_titanic_clean = df_titanic[['Survived', 'Pclass', 'Sex', 'Age', 'Embarked']].copy()

    # Gestion des valeurs manquantes
    df_titanic_clean['Age'].fillna(df_titanic_clean['Age'].median(), inplace=True)
    df_titanic_clean['Embarked'].fillna('S', inplace=True)  # Mode
    df_titanic_clean = df_titanic_clean.dropna()

    # Encodage (comme dans le modèle)
    df_titanic_clean['sex'] = df_titanic_clean['Sex'].map({'male': 0, 'female': 1})
    df_titanic_clean['pclass'] = df_titanic_clean['Pclass']
    df_titanic_clean['age'] = df_titanic_clean['Age']
    df_titanic_clean['embarked'] = df_titanic_clean['Embarked'].map({'C': 0, 'Q': 1, 'S': 2})
    df_titanic_clean['survived'] = df_titanic_clean['Survived']

    # Dataset final pour référence
    reference_titanic = df_titanic_clean[['sex', 'pclass', 'age', 'embarked', 'survived']].copy()
    reference_titanic['target'] = reference_titanic['survived']  # Alias pour Evidently

    # Split train/test (80/20 comme dans le notebook unemployment)
    train_titanic, test_titanic = train_test_split(
        reference_titanic, test_size=0.2, random_state=42, stratify=reference_titanic['survived']
    )

    # Sauvegarder le train comme référence
    output_file = OUTPUT_DIR / "titanic_reference.csv"
    train_titanic.to_csv(output_file, index=False)
    print(f"[OK] Titanic reference saved: {train_titanic.shape}")
    print(f"   Fichier: {output_file}")
    print(f"   Features: sex, pclass, age, embarked")
    print(f"   Taux de survie: {train_titanic['survived'].mean():.2%}")
    print(f"   Distribution sexe: {train_titanic['sex'].value_counts().to_dict()}")
    print(f"   Distribution classe: {train_titanic['pclass'].value_counts().to_dict()}")
else:
    print(f"[ERROR] Fichier non trouve: {titanic_file}")


# ===== 2. UNEMPLOYMENT REFERENCE DATA =====
print("\n[UNEMPLOYMENT] Extraction donnees Unemployment...")

unemployment_file = ML_DATA_DIR / "Employment_Unemployment_GDP_data.csv"

if unemployment_file.exists():
    df_unemployment = pd.read_csv(unemployment_file)

    # Renommer les colonnes (comme dans le notebook)
    df_unemployment.rename(columns={
        'Country Name': 'country',
        'Year': 'year',
        'Employment Sector: Agriculture': 'agriculture',
        'Employment Sector: Industry': 'industry',
        'Employment Sector: Services': 'services',
        'Unemployment Rate': 'unemployment_rate',
        'GDP (in USD)': 'gdp'
    }, inplace=True)

    # Ajouter gdp_log
    df_unemployment['gdp_log'] = np.log1p(df_unemployment['gdp'])

    # Utiliser 80% comme référence (même split que l'entraînement)
    features = ['agriculture', 'industry', 'services', 'gdp_log', 'year', 'country']
    target = 'unemployment_rate'

    X = df_unemployment[features]
    y = df_unemployment[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Combiner features + target pour référence
    reference_unemployment = X_train.copy()
    reference_unemployment['unemployment_rate'] = y_train
    reference_unemployment['target'] = y_train  # Alias pour Evidently

    # Sauvegarder
    output_file = OUTPUT_DIR / "unemployment_reference.csv"
    reference_unemployment.to_csv(output_file, index=False)
    print(f"[OK] Unemployment reference saved: {reference_unemployment.shape}")
    print(f"   Fichier: {output_file}")
    print(f"   Features: {features}")
    print(f"   Periode: {reference_unemployment['year'].min()}-{reference_unemployment['year'].max()}")
    print(f"   Pays: {reference_unemployment['country'].nunique()} pays")
    print(f"   Chomage moyen: {reference_unemployment['unemployment_rate'].mean():.2f}%")
else:
    print(f"[ERROR] Fichier non trouve: {unemployment_file}")


print("\n[SUCCESS] Extraction terminee ! Les datasets de reference sont prets.")
print(f"\n[FILES] Fichiers crees:")
print(f"   - {OUTPUT_DIR / 'titanic_reference.csv'}")
print(f"   - {OUTPUT_DIR / 'unemployment_reference.csv'}")
print("\n[INFO] Ces fichiers serviront de baseline pour detecter le drift avec Evidently.")
