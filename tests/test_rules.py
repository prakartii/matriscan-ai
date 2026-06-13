# tests/test_rules.py
import pytest
from app.engine.rules import run_rules_engine
from app.schemas.models import VitalSigns, Symptoms

def make_vitals(**kwargs):
    return VitalSigns(**kwargs)

def make_symptoms(**kwargs):
    return Symptoms(**kwargs)

def test_kavitha_rajan_fires_red():
    """Flagship demo case must always be RED."""
    vitals = make_vitals(
        bp_systolic=148, bp_diastolic=96,
        hemoglobin=8.2, urine_protein="2+"
    )
    symptoms = make_symptoms(headache=True, blurred_vision=True)
    result = run_rules_engine(vitals, symptoms)
    assert result["risk_color"] == "RED"
    
def test_severe_bp_fires_red():
    vitals = make_vitals(bp_systolic=170, bp_diastolic=115)
    symptoms = make_symptoms()
    result = run_rules_engine(vitals, symptoms)
    assert result["risk_color"] == "RED"

def test_severe_anemia_fires_red():
    vitals = make_vitals(hemoglobin=5.5)
    symptoms = make_symptoms()
    result = run_rules_engine(vitals, symptoms)
    assert result["risk_color"] == "RED"

def test_moderate_anemia_fires_amber():
    vitals = make_vitals(hemoglobin=8.5)
    symptoms = make_symptoms()
    result = run_rules_engine(vitals, symptoms)
    assert result["risk_color"] == "AMBER"

def test_normal_patient_fires_green():
    vitals = make_vitals(
        bp_systolic=118, bp_diastolic=76,
        hemoglobin=12.5,
        glucose_fasting=85, glucose_pp=110
    )
    symptoms = make_symptoms()
    result = run_rules_engine(vitals, symptoms)
    assert result["risk_color"] == "GREEN"

def test_gdm_elevated_glucose_fires_amber():
    vitals = make_vitals(glucose_fasting=105, glucose_pp=145)
    symptoms = make_symptoms()
    result = run_rules_engine(vitals, symptoms)
    assert result["risk_color"] in ["AMBER", "RED"]

def test_sepsis_multiple_indicators_fires_red():
    vitals = make_vitals(temperature=101.2, pulse=108)
    symptoms = make_symptoms(fever=True, dysuria=True, foul_discharge=True)
    result = run_rules_engine(vitals, symptoms)
    assert result["risk_color"] == "RED"

def test_missing_vitals_does_not_crash():
    vitals = make_vitals()
    symptoms = make_symptoms()
    result = run_rules_engine(vitals, symptoms)
    assert "risk_color" in result
    assert result["risk_color"] == "GREEN"