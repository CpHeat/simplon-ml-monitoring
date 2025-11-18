# app/model_loader.py
from joblib import load
from pathlib import Path
from tensorflow import keras  # ✅ Import TensorFlow
# Point d'accès racine du projet
app_path = Path(__file__).resolve().parent  # Remonte au dossier fast_api/

# Chemins absolus
MODEL_DIR = app_path / "models"  # fast_api/models/
MODEL_ML_PATH = MODEL_DIR / "ml_model.pkl"
SCALER_ML_PATH = MODEL_DIR / "ml_scaler.pkl"

MODEL_DL_PATH = MODEL_DIR / "dl_model_best.keras"  # Utilise le meilleur modèle
SCALER_DL_PATH = MODEL_DIR / "dl_scaler.pkl"
LE_SEX_PATH = MODEL_DIR / "dl_le_sex.pkl"
LE_EMBARKED_PATH = MODEL_DIR / "dl_le_embarked.pkl"

# Vérification
if not MODEL_ML_PATH.exists():
    raise FileNotFoundError(f" Fichier introuvable : {MODEL_ML_PATH}")
if not SCALER_ML_PATH.exists():
    raise FileNotFoundError(f"Fichier introuvable : {SCALER_ML_PATH}")

# Vérification DL
if not MODEL_DL_PATH.exists():
    raise FileNotFoundError(f"Fichier introuvable : {MODEL_DL_PATH}")
if not SCALER_DL_PATH.exists():
    raise FileNotFoundError(f"Fichier introuvable : {SCALER_DL_PATH}")
if not LE_SEX_PATH.exists():
    raise FileNotFoundError(f"Fichier introuvable : {LE_SEX_PATH}")
if not LE_EMBARKED_PATH.exists():
    raise FileNotFoundError(f"Fichier introuvable : {LE_EMBARKED_PATH}")

# Chargement ML
model_titanic_ml = load(MODEL_ML_PATH)
scaler_X_titanic_ml = load(SCALER_ML_PATH)

# Chargement DL
model_titanic_dl = keras.models.load_model(MODEL_DL_PATH)
scaler_X_titanic_dl = load(SCALER_DL_PATH)
le_sex = load(LE_SEX_PATH)
le_embarked = load(LE_EMBARKED_PATH)