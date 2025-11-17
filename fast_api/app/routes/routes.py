from fastapi import APIRouter,HTTPException,Query
from app.schemas import InputMachine, InputDeep
# from app.predict import Predict

router = APIRouter(prefix="/predict")
# predict = Predict()

@router.post("/ml",summary="Prediction si en fonction du genre taux de chance de survie machine learning",
           description="connaitre le taux de chance de survie en fonction de différent parametre")
async def predict_survibality_machine(data: InputMachine,genre: str = Query(..., description="homme ou femme")):
    """
    Renvoie la prédiction de taux de survie en fonction de différents parametre.
    
    Returns:
        float: pourcentage de survie
        
    Raises:
        HTTPException: Si des données manquantes
    """
    genre=genre.lower()
    rate_survive=0
    try:
        if(genre=="homme"):
            rate_survive=25
        elif(genre=="femme"):
            rate_survive=75
        else:
           raise HTTPException(status_code=400, detail=str(e)) 
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    
    return {
        "rate":rate_survive
    }

@router.post("/dl",summary="Prediction si en fonction du genre taux de chance de survie deep learning",
           description="connaitre le taux de chance de survie en fonction de différent parametre")
async def predict_survibality_deep(data: InputDeep, genre: str = Query(..., description="homme ou femme")):
    """
    Renvoie la prédiction de taux de survie en fonction de différents parametre.
    
    Returns:
        float: pourcentage de survie
        
    Raises:
        HTTPException: Si des données manquantes
    """
    genre=genre.lower()
    rate_survive=0
    try:
        if(genre=="homme"):
            rate_survive=20
        elif(genre=="femme"):
            rate_survive=80
        else:
           raise HTTPException(status_code=400, detail=str(e)) 
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    
    return {
        "rate":rate_survive
    }