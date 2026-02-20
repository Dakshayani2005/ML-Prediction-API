from pydantic import BaseModel
from typing import List

class PredictionResponse(BaseModel):
    class_label: str
    probabilities: List[float]