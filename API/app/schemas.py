from pydantic import BaseModel,Field
from typing import Optional,Literal

class InputMachine(BaseModel):
    genre : Literal["homme", "femme"]
    pclass : int = Field(..., ge=1, le=3)
    age : float = Field(..., ge=0, le=80)

class InputDeep(BaseModel):
    """Input pour Deep Learning (plus de features)"""
    genre: Literal["homme", "femme"]
    pclass: int = Field(..., ge=1, le=3)
    age: int = Field(..., ge=0, le=80)
    embarked: Literal["C", "Q", "S"] = Field(..., description="Port d'embarquement")

# ===== Unemployment Schema =====
class InputUnemployment(BaseModel):
    """Input pour prédiction taux de chômage"""
    country: str = Field(..., description="Nom du pays (ex: France, USA, etc.)")
    agriculture: float = Field(..., ge=0, le=100, description="Part agriculture dans PIB (%)")
    industry: float = Field(..., ge=0, le=100, description="Part industrie dans PIB (%)")
    services: float = Field(..., ge=0, le=100, description="Part services dans PIB (%)")
    gdp_log: float = Field(..., description="Log du PIB")
    year: int = Field(..., ge=1960, le=2030, description="Année")
    
    class Config:
        json_schema_extra = {
            "example": {
                "country": "France",
                "agriculture": 2.5,
                "industry": 19.8,
                "services": 77.7,
                "gdp_log": 27.5,
                "year": 2023
            }
        }

class UnemploymentResponse(BaseModel):
    """Réponse de prédiction chômage"""
    predicted_unemployment_rate: float = Field(..., description="Taux de chômage prédit (%)")
    input_data: dict
    model: str = Field(default="XGBoost")