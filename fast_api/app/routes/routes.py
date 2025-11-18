from fastapi import APIRouter,HTTPException,Query
from app.schemas import InputMachine, InputDeep
from app.predict import Predict

router = APIRouter(prefix="/predict")
predictor = Predict()

@router.post("/ml")
async def predict_ml(data: InputMachine):
    try:
        result = await predictor.predict_survive_ml(
            data=data.dict(),
            genre=data.genre,
            pclass=data.pclass,
            age=data.age
        )
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))