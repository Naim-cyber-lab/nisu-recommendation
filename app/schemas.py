from typing import List, Optional
from datetime import date, datetime
from pydantic import BaseModel


# ---------- WINKER ----------

class WinkerIn(BaseModel):
    id: int
    username: str
    email: Optional[str] = None
    sexe: Optional[str] = None          # "Masculin" / "Feminin" / etc.
    age: Optional[int] = None           # calculé à partir de birth / birthYear côté Django
    city: Optional[str] = None
    region: Optional[str] = None
    subregion: Optional[str] = None
    pays: Optional[str] = None
    lat: Optional[float] = None
    lon: Optional[float] = None
    visible_tags: List[str] = []        # dérivées de VisiblePreferenceWinker.party/sport/...
    preference_vector: Optional[List[float]] = None  # len 16
    meet_eligible: Optional[bool] = None
    mails_eligible: Optional[bool] = None


class WinkerOut(WinkerIn):
    score: Optional[float] = None


# ---------- EVENT ----------

class EventIn(BaseModel):
    id: int
    titre: Optional[str] = None
    bioEvent: Optional[str] = None
    city: Optional[str] = None
    region: Optional[str] = None
    subregion: Optional[str] = None
    pays: Optional[str] = None
    codePostal: Optional[str] = None
    lat: Optional[float] = None
    lon: Optional[float] = None
    dateEvent: Optional[date] = None
    datePublication: Optional[date] = None
    ageMinimum: Optional[int] = None
    ageMaximum: Optional[int] = None
    accessFille: Optional[bool] = None
    accessGarcon: Optional[bool] = None
    accessTous: Optional[bool] = None
    hastagEvents: List[str] = []              # hashtags déjà splittés côté Django
    meetEligible: Optional[bool] = None
    planTripElligible: Optional[bool] = None
    currentNbParticipants: Optional[int] = None
    maxNumberParticipant: Optional[int] = None
    isFull: Optional[bool] = None
    vectorPreferenceEvent: Optional[List[float]] = None  # len 16


class EventOut(EventIn):
    score: Optional[float] = None


# ---------- RECO OUTPUT ----------

class RecommendedWinker(BaseModel):
    winker: WinkerOut
    score: float


class RecommendedEvent(BaseModel):
    event: EventOut
    score: float
