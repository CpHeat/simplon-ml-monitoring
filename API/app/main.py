from prometheus_client import REGISTRY
from datetime import datetime
from fastapi import FastAPI
from prometheus_fastapi_instrumentator import Instrumentator
from app.model_loader import model_titanic_ml, model_titanic_dl
from .routes.routes import router

#python -m uvicorn app.main:app --reload --port 8000

app = FastAPI(
    title="API prediction pour le dataset du titanic pour connaitre les différents parametre qui a permets de survivre",
    description="API pour la prédiction du titanic",
)

Instrumentator().instrument(app).expose(app)

app.include_router(router)

@app.get("/")
def root():
    return {"message": "Titanic Prediction API"}

@app.get("/health", tags=["Monitoring"])
def health_check():
    """Endpoint pour vérifier que l'API est en ligne"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "1.0.0"
    }