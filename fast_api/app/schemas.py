from pydantic import BaseModel,Field
from typing import Optional,Literal

class InputMachine(BaseModel):
    genre : Literal["homme", "femme"]

class InputDeep(BaseModel):
    genre : Literal["homme", "femme"]
    passenger_class : int = Field(..., ge=1, le=3)
    age : int = Field(..., ge=0, le=100)