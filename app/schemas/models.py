# app/schemas/models.py
from pydantic import BaseModel
from typing import Optional, List, Dict, Any

# --- Visit data shapes ---

class VitalSigns(BaseModel):
    bp_systolic: Optional[float] = None
    bp_diastolic: Optional[float] = None
    hemoglobin: Optional[float] = None
    weight: Optional[float] = None
    pulse: Optional[float] = None
    temperature: Optional[float] = None
    spo2: Optional[float] = None
    glucose_fasting: Optional[float] = None
    glucose_pp: Optional[float] = None
    urine_protein: Optional[str] = None   # "negative", "1+", "2+", "3+"
    platelets: Optional[float] = None

class Symptoms(BaseModel):
    headache: bool = False
    blurred_vision: bool = False
    epigastric_pain: bool = False
    fever: bool = False
    reduced_fetal_movement: bool = False
    vaginal_bleeding: bool = False
    swelling: bool = False
    dysuria: bool = False                 # painful urination (sepsis indicator)
    foul_discharge: bool = False

class VisitRecord(BaseModel):
    visit_id: str
    patient_id: str
    visit_number: int
    gestational_age: int                  # weeks
    vitals: VitalSigns
    symptoms: Symptoms
    visit_date: Optional[str] = None      # ISO date string "2024-01-15"

class PatientInfo(BaseModel):
    patient_id: str
    age: int
    parity: str                           # e.g. "G2P1" = 2nd pregnancy, 1 delivery
    language: str = "ta"                  # default Tamil

# --- Request bodies for each endpoint ---

class AnalyzeRequest(BaseModel):
    patient: PatientInfo
    current_visit: VisitRecord
    visit_history: List[VisitRecord] = [] # last 5 visits for trend analysis

class CarePlanRequest(BaseModel):
    patient: PatientInfo
    risk_result: Dict[str, Any]           # output from /ai/analyze
    language: str = "ta"

class TranscribeRequest(BaseModel):
    language: str = "ta"                  # language code for Whisper

class TTSRequest(BaseModel):
    text: str
    language: str = "ta-IN"

# --- Response shapes ---

class RiskResponse(BaseModel):
    overall_risk: str                     # "HIGH", "MODERATE", "LOW"
    risk_color: str                       # "RED", "AMBER", "GREEN"
    matri_score: int                      # 0-100
    confidence: float                     # 0.0-1.0
    domain_scores: Dict[str, float]
    trend_flags: List[str]
    flagged_values: List[str]
    explanations: List[str]

class CarePlanResponse(BaseModel):
    doctor_advice: str
    tamil_counseling: str
    followup_days: int
    referral_urgency: str
    red_flags: List[str]
    audio_url: Optional[str] = None