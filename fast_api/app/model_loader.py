# app/model_loader.py
from joblib import load
from pathlib import Path

# Point d'accès racine du projet
app_path = Path(__file__).resolve().parent  # Remonte au dossier fast_api/

# Chemins absolus
MODEL_DIR = app_path / "models"  # fast_api/models/
MODEL_ML_PATH = MODEL_DIR / "ml_model.pkl"
SCALER_ML_PATH = MODEL_DIR / "ml_scaler.pkl"

print(f"📁 root_path : {app_path}")
print(f"📁 MODEL_DIR : {MODEL_DIR}")
print(f"📄 MODEL_ML_PATH : {MODEL_ML_PATH}")

# Vérification
if not MODEL_ML_PATH.exists():
    raise FileNotFoundError(f"❌ Fichier introuvable : {MODEL_ML_PATH}")
if not SCALER_ML_PATH.exists():
    raise FileNotFoundError(f"❌ Fichier introuvable : {SCALER_ML_PATH}")

# Chargement
model_titanic_ml = load(MODEL_ML_PATH)
scaler_X_titanic_ml = load(SCALER_ML_PATH)

print(f"✅ Type modèle : {type(model_titanic_ml)}")
print(f"✅ Type scaler : {type(scaler_X_titanic_ml)}")