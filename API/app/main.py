from fastapi import FastAPI
from prometheus_fastapi_instrumentator import Instrumentator
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

@app.get("/health")
def health():
    return {"status": "healthy"}