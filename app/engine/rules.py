# app/engine/rules.py
from typing import Dict, Any, Optional
from app.schemas.models import VitalSigns, Symptoms

# ── Severity constants ──────────────────────────────────────────────
SEVERITY_NONE     = 0
SEVERITY_MILD     = 1
SEVERITY_MODERATE = 2
SEVERITY_SEVERE   = 3

def _protein_numeric(value: Optional[str]) -> int:
    """Convert urine protein string to numeric for comparison."""
    mapping = {"negative": 0, "trace": 0, "1+": 1, "2+": 2, "3+": 3, "4+": 4}
    if value is None:
        return -1   # -1 = not assessed
    return mapping.get(value.lower().strip(), -1)


# ── Individual condition checkers ───────────────────────────────────

def check_hypertension(vitals: VitalSigns, symptoms: Symptoms) -> Dict[str, Any]:
    """
    WHO/ACOG preeclampsia criteria.
    Severe features: BP >= 160/110, OR severe neurological symptoms
    (headache + blurred vision = severe preeclampsia regardless of exact BP)
    """
    result = {
        "condition": "hypertension_preeclampsia",
        "severity": SEVERITY_NONE,
        "flags": [],
        "score": 0.0
    }

    bp_s = vitals.bp_systolic
    bp_d = vitals.bp_diastolic
    protein = _protein_numeric(vitals.urine_protein)

    if bp_s is None and bp_d is None:
        result["flags"].append("BP not assessed")
        return result

    # Severe preeclampsia by BP threshold
    if (bp_s is not None and bp_s >= 160) or (bp_d is not None and bp_d >= 110):
        result["severity"] = SEVERITY_SEVERE
        result["score"] = 0.95
        result["flags"].append(f"Severe hypertension: BP {bp_s}/{bp_d} (threshold 160/110)")

    # Severe preeclampsia by neurological symptoms (ACOG severe features)
    # Headache + blurred vision = severe preeclampsia even if BP hasn't hit 160
    elif symptoms.headache and symptoms.blurred_vision:
        result["severity"] = SEVERITY_SEVERE
        result["score"] = 0.90
        result["flags"].append("Severe preeclampsia features: headache + blurred vision")
        if bp_s is not None and bp_s >= 140:
            result["flags"].append(f"BP {bp_s}/{bp_d} confirms hypertensive range")

    # Moderate preeclampsia: BP >= 140/90 + protein
    elif (bp_s is not None and bp_s >= 140) or (bp_d is not None and bp_d >= 90):
        if protein >= 2:
            result["severity"] = SEVERITY_MODERATE
            result["score"] = 0.75
            result["flags"].append(f"Preeclampsia: BP {bp_s}/{bp_d} + protein {vitals.urine_protein}")
        elif protein == 1:
            result["severity"] = SEVERITY_MILD
            result["score"] = 0.55
            result["flags"].append(f"Gestational hypertension: BP {bp_s}/{bp_d}, trace protein")
        else:
            result["severity"] = SEVERITY_MILD
            result["score"] = 0.45
            result["flags"].append(f"Elevated BP: {bp_s}/{bp_d} — monitor closely")

    # Epigastric pain is a severe feature even with moderate BP
    if symptoms.epigastric_pain and result["severity"] >= SEVERITY_MODERATE:
        result["severity"] = SEVERITY_SEVERE
        result["score"] = min(result["score"] + 0.10, 1.0)
        result["flags"].append("Epigastric pain: severe preeclampsia feature (HELLP risk)")

    return result


def check_anemia(vitals: VitalSigns) -> Dict[str, Any]:
    """
    WHO anemia in pregnancy thresholds.
    Severe < 7, Moderate 7-9.9, Mild 10-10.9
    """
    result = {
        "condition": "anemia",
        "severity": SEVERITY_NONE,
        "flags": [],
        "score": 0.0
    }

    hb = vitals.hemoglobin
    if hb is None:
        result["flags"].append("Hemoglobin not assessed")
        return result

    if hb < 7.0:
        result["severity"] = SEVERITY_SEVERE
        result["score"] = 0.95
        result["flags"].append(f"Severe anemia: Hb {hb} g/dL (threshold < 7.0)")
    elif hb < 10.0:
        result["severity"] = SEVERITY_MODERATE
        result["score"] = 0.65
        result["flags"].append(f"Moderate anemia: Hb {hb} g/dL (threshold 7-10)")
    elif hb < 11.0:
        result["severity"] = SEVERITY_MILD
        result["score"] = 0.35
        result["flags"].append(f"Mild anemia: Hb {hb} g/dL (threshold 10-11)")
    else:
        result["flags"].append(f"Hb {hb} g/dL — normal range")

    return result


def check_gdm(vitals: VitalSigns) -> Dict[str, Any]:
    """
    IADPSG/WHO gestational diabetes thresholds.
    Fasting > 92 mg/dL, Post-prandial > 120 mg/dL
    """
    result = {
        "condition": "gestational_diabetes",
        "severity": SEVERITY_NONE,
        "flags": [],
        "score": 0.0
    }

    flags_found = []

    if vitals.glucose_fasting is not None:
        if vitals.glucose_fasting > 126:
            result["severity"] = max(result["severity"], SEVERITY_SEVERE)
            result["score"] = max(result["score"], 0.90)
            flags_found.append(f"Critical fasting glucose: {vitals.glucose_fasting} mg/dL")
        elif vitals.glucose_fasting > 92:
            result["severity"] = max(result["severity"], SEVERITY_MODERATE)
            result["score"] = max(result["score"], 0.70)
            flags_found.append(f"Elevated fasting glucose: {vitals.glucose_fasting} mg/dL (threshold > 92)")

    if vitals.glucose_pp is not None:
        if vitals.glucose_pp > 200:
            result["severity"] = max(result["severity"], SEVERITY_SEVERE)
            result["score"] = max(result["score"], 0.92)
            flags_found.append(f"Critical post-prandial glucose: {vitals.glucose_pp} mg/dL")
        elif vitals.glucose_pp > 120:
            result["severity"] = max(result["severity"], SEVERITY_MODERATE)
            result["score"] = max(result["score"], 0.68)
            flags_found.append(f"Elevated PP glucose: {vitals.glucose_pp} mg/dL (threshold > 120)")

    if not flags_found:
        if vitals.glucose_fasting is None and vitals.glucose_pp is None:
            result["flags"].append("Glucose not assessed")
        else:
            result["flags"].append("Glucose within normal range")
    else:
        result["flags"].extend(flags_found)

    return result


def check_sepsis(vitals: VitalSigns, symptoms: Symptoms) -> Dict[str, Any]:
    """
    Sepsis risk: fever + tachycardia + infection symptoms.
    Two or more infection indicators = HIGH risk.
    """
    result = {
        "condition": "sepsis_risk",
        "severity": SEVERITY_NONE,
        "flags": [],
        "score": 0.0
    }

    infection_indicators = []

    if vitals.temperature is not None and vitals.temperature > 100.4:
        infection_indicators.append(f"Fever: {vitals.temperature}°F")

    if vitals.pulse is not None and vitals.pulse > 100:
        infection_indicators.append(f"Tachycardia: pulse {vitals.pulse} bpm")

    if symptoms.fever:
        infection_indicators.append("Reported fever")

    if symptoms.dysuria:
        infection_indicators.append("Dysuria (UTI indicator)")

    if symptoms.foul_discharge:
        infection_indicators.append("Foul vaginal discharge")

    count = len(infection_indicators)

    if count >= 3:
        result["severity"] = SEVERITY_SEVERE
        result["score"] = 0.88
        result["flags"].extend(infection_indicators)
        result["flags"].append("HIGH sepsis risk — multiple indicators")
    elif count == 2:
        result["severity"] = SEVERITY_MODERATE
        result["score"] = 0.60
        result["flags"].extend(infection_indicators)
    elif count == 1:
        result["severity"] = SEVERITY_MILD
        result["score"] = 0.30
        result["flags"].extend(infection_indicators)
    else:
        result["flags"].append("No sepsis indicators")

    return result


# ── Master rules runner ─────────────────────────────────────────────

def run_rules_engine(vitals: VitalSigns, symptoms: Symptoms) -> Dict[str, Any]:
    """
    Run all condition checks and aggregate into a single risk assessment.
    Returns domain scores, highest severity, all flags, and explanations.
    """
    htn   = check_hypertension(vitals, symptoms)
    anemia = check_anemia(vitals)
    gdm   = check_gdm(vitals)
    sepsis = check_sepsis(vitals, symptoms)

    domains = {
        "preeclampsia": htn,
        "anemia": anemia,
        "gdm": gdm,
        "sepsis": sepsis
    }

    # Highest severity across all domains determines overall risk
    max_severity = max(d["severity"] for d in domains.values())

    # Collect all meaningful flags
    all_flags = []
    all_explanations = []
    for domain_name, domain_result in domains.items():
        for flag in domain_result["flags"]:
            if "not assessed" not in flag.lower() and "normal" not in flag.lower():
                all_flags.append(flag)
                all_explanations.append(f"[{domain_name.upper()}] {flag}")

    # Map severity to risk level and color
    if max_severity >= SEVERITY_SEVERE:
        overall_risk = "HIGH"
        risk_color   = "RED"
        base_score   = 80
    elif max_severity == SEVERITY_MODERATE:
        overall_risk = "MODERATE"
        risk_color   = "AMBER"
        base_score   = 55
    elif max_severity == SEVERITY_MILD:
        overall_risk = "LOW-MODERATE"
        risk_color   = "AMBER"
        base_score   = 35
    else:
        overall_risk = "LOW"
        risk_color   = "GREEN"
        base_score   = 10

    # Matri score: base + weighted domain contributions
    domain_scores = {name: round(d["score"], 2) for name, d in domains.items()}
    return {
        "overall_risk":   overall_risk,
        "risk_color":     risk_color,
        "domain_scores":  domain_scores,
        "max_severity":   max_severity,
        "flagged_values": all_flags,
        "explanations":   all_explanations,
        "raw_domains":    domains
    }
