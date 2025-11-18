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