"""
Module pour la gestion de la base de données SQLite.
Stocke toutes les prédictions pour l'analyse de drift avec Evidently.
"""
import os
from sqlalchemy import create_engine, Column, Integer, Float, String, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime
from pathlib import Path

# Configuration base de données
# Support pour Docker et développement local
DB_PATH_ENV = os.getenv("DB_PATH")

if DB_PATH_ENV:
    # En production Docker : utilise la variable d'environnement
    DB_PATH = Path(DB_PATH_ENV)
else:
    # En développement local : utilise le chemin relatif
    DB_DIR = Path(__file__).parent.parent.parent / "MONITORING" / "evidently" / "data"
    DB_DIR.mkdir(parents=True, exist_ok=True)
    DB_PATH = DB_DIR / "predictions.db"

# Créer le dossier si nécessaire
DB_PATH.parent.mkdir(parents=True, exist_ok=True)

DATABASE_URL = f"sqlite:///{DB_PATH}"

engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


# ===== MODÈLES DE TABLES =====

class TitanicPrediction(Base):
    """Table pour les prédictions Titanic (ML et DL)"""
    __tablename__ = "titanic_predictions"

    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)
    model_type = Column(String)  # 'ml' ou 'dl'

    # Features
    sex = Column(Integer)  # 0=homme, 1=femme
    pclass = Column(Integer)  # 1, 2, 3
    age = Column(Float)
    embarked = Column(Integer, nullable=True)  # 0=C, 1=Q, 2=S (seulement pour DL)

    # Prédictions
    prediction = Column(Integer)  # 0=décès, 1=survie
    probability_survive = Column(Float)
    probability_death = Column(Float)

    # Vérité terrain (optionnel, pour concept drift)
    true_label = Column(Integer, nullable=True)  # 0=décès, 1=survie


class UnemploymentPrediction(Base):
    """Table pour les prédictions de taux de chômage"""
    __tablename__ = "unemployment_predictions"

    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)

    # Features
    country = Column(String, index=True)
    year = Column(Integer)
    agriculture = Column(Float)
    industry = Column(Float)
    services = Column(Float)
    gdp_log = Column(Float)

    # Prédiction
    predicted_unemployment_rate = Column(Float)

    # Pas de true_label pour unemployment (pas disponible en production)


# Créer toutes les tables
Base.metadata.create_all(bind=engine)


# ===== FONCTIONS HELPER =====

def get_db():
    """Générateur de session DB pour FastAPI dependency injection"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def get_db_session():
    """Obtenir une session DB standalone"""
    return SessionLocal()


# Logs
print(f"[DATABASE] Initialise: {DB_PATH}")
print(f"[DATABASE] Tables creees: {Base.metadata.tables.keys()}")
