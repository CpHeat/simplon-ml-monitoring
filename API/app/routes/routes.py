from fastapi import APIRouter,HTTPException,Query
from app.schemas import (InputMachine, InputDeep,InputUnemployment,UnemploymentResponse)
from app.predict import Predict
from app.data_logger import data_logger

router = APIRouter(prefix="/predict")
predictor = Predict()

@router.post("/titanic_ml", summary="Prédiction Machine Learning - Survie Titanic")
async def predict_ml(data: InputMachine):
    try:
        result = await predictor.predict_survive_ml(
            data=data.dict(),
            genre=data.genre,
            pclass=data.pclass,
            age=data.age
        )

        # Logger la prédiction pour le monitoring de drift
        sex_encoded = 1 if data.genre == "femme" else 0
        await data_logger.log_titanic_prediction(
            model_type="ml",
            sex=sex_encoded,
            pclass=data.pclass,
            age=data.age,
            prediction=result["prediction"],
            probability_survive=result["probability_survive"],
            probability_death=result["probability_death"],
            embarked=None,  # ML model n'utilise pas embarked
            true_label=data.true_label
        )

        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@router.post("/titanic_dl", summary="Prédiction Deep Learning - Survie Titanic")
async def predict_dl(data: InputDeep):
    try:
        result = await predictor.predict_survive_dl(
            data=data.dict(),
            genre=data.genre,
            pclass=data.pclass,
            age=data.age,
            embarked=data.embarked
        )

        # Logger la prédiction pour le monitoring de drift
        sex_encoded = 1 if data.genre == "femme" else 0
        embarked_mapping = {"C": 0, "Q": 1, "S": 2}
        embarked_encoded = embarked_mapping[data.embarked]

        await data_logger.log_titanic_prediction(
            model_type="dl",
            sex=sex_encoded,
            pclass=data.pclass,
            age=data.age,
            prediction=result["prediction"],
            probability_survive=result["probability_survive"],
            probability_death=result["probability_death"],
            embarked=embarked_encoded,
            true_label=data.true_label
        )

        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
# ===== XGBoost Unemployment =====
@router.post("/unemployment", 
             response_model=UnemploymentResponse,
             summary="Prédiction Taux de Chômage")
async def predict_unemployment(data: InputUnemployment):
    """
    Prédit le taux de chômage en fonction des indicateurs économiques.
    
    Features utilisées:
    - agriculture: Part du secteur agricole dans le PIB (%)
    - industry: Part du secteur industriel dans le PIB (%)
    - services: Part du secteur des services dans le PIB (%)
    - gdp_log: Logarithme du PIB
    - year: Année de prédiction
    - country: Pays
    """
    try:
        result = await predictor.predict_unemployment(
            country=data.country,
            agriculture=data.agriculture,
            industry=data.industry,
            services=data.services,
            gdp_log=data.gdp_log,
            year=data.year
        )

        # Logger la prédiction pour le monitoring de drift
        await data_logger.log_unemployment_prediction(
            country=data.country,
            year=data.year,
            agriculture=data.agriculture,
            industry=data.industry,
            services=data.services,
            gdp_log=data.gdp_log,
            predicted_unemployment_rate=result["predicted_unemployment_rate"]
        )

        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))