from pydantic import BaseModel
from typing import List


class SpillResult(BaseModel):
    confidence: float
    area_km2: float


class OriginResult(BaseModel):
    latitude: float
    longitude: float


class VesselResult(BaseModel):
    id: str
    score: float
    reasons: List[str]

class InvestigationResult(BaseModel):
    spill: SpillResult
    origin: OriginResult
    vessels: List[VesselResult]

class InvestigationRequest(BaseModel):
    latitude: float
    longitude: float
    timestamp: str
    image_path: str