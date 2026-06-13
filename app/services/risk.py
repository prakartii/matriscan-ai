# app/services/risk.py
from typing import List, Dict, Any
import pandas as pd
import numpy as np
from app.engine.rules import run_rules_engine, SEVERITY_SEVERE, SEVERITY_MODERATE
from app.schemas.models import VisitRecord, PatientInfo

# Slope threshold: if BP rising > 2 mmHg/week, flag it
BP_RISING_SLOPE_THRESHOLD  =  2.0
BP_FALLING_SLOPE_THRESHOLD = -2.0
HB_FALLING_SLOPE_THRESHOLD = -0.3   # g/dL per week


def _compute_trends(visit_history: List[VisitRecord]) -> Dict[str, Any]:
    """
    Linear regression on vitals across visits.
    Returns slope values and human-readable trend flags.
    """
    trend_flags = []
    trend_details = {}

    if len(visit_history) < 2:
        return {"trend_flags": [], "trend_details": {}, "score_boost": 0}

    # Build a DataFrame from visit history
    records = []
    for v in visit_history:
        records.append({
            "visit_number":  v.visit_number,
            "bp_systolic":   v.vitals.bp_systolic,
            "bp_diastolic":  v.vitals.bp_diastolic,
            "hemoglobin":    v.vitals.hemoglobin,
        })

    df = pd.DataFrame(records).sort_values("visit_number")

    # x-axis: visit number (proxy for time)
    x = df["visit_number"].values.astype(float)

    score_boost = 0

    # BP systolic trend
    bp_vals = df["bp_systolic"].dropna()
    if len(bp_vals) >= 2:
        idx = df["bp_systolic"].dropna().index
        slope = float(np.polyfit(x[df.index.isin(idx)], bp_vals.values, 1)[0])
        trend_details["bp_systolic_slope"] = round(slope, 2)

        if slope > BP_RISING_SLOPE_THRESHOLD:
            trend_flags.append("BP_RISING")
            score_boost += 8
        elif slope < BP_FALLING_SLOPE_THRESHOLD:
            trend_flags.append("BP_FALLING")

    # Hemoglobin trend
    hb_vals = df["hemoglobin"].dropna()
    if len(hb_vals) >= 2:
        idx = df["hemoglobin"].dropna().index
        slope = float(np.polyfit(x[df.index.isin(idx)], hb_vals.values, 1)[0])
        trend_details["hb_slope"] = round(slope, 2)

        if slope < HB_FALLING_SLOPE_THRESHOLD:
            trend_flags.append("HB_FALLING")
            score_boost += 6

    return {
        "trend_flags":   trend_flags,
        "trend_details": trend_details,
        "score_boost":   score_boost
    }


def run_full_analysis(
    patient: PatientInfo,
    current_visit: VisitRecord,
    visit_history: List[VisitRecord]
) -> Dict[str, Any]:
    """
    Full pipeline:
    1. WHO rules engine on current visit
    2. Trend analysis on visit history
    3. Combine into final risk response
    Rules layer can never be overridden downward.
    """

    # Step 1: Rules engine
    rules_result = run_rules_engine(
        current_visit.vitals,
        current_visit.symptoms
    )

    # Step 2: Trend analysis
    # Include current visit in history for trend computation
    full_history = visit_history + [current_visit]
    trend_result = _compute_trends(full_history)

    # Step 3: Combine
    # Boost matri_score by trend findings, cap at 100
    final_score = min(
        rules_result["matri_score"] + trend_result["score_boost"],
        100
    )

    # Confidence: higher when more visit history available
    # Confidence label based on history depth — this is honest and explainable
    history_count = len(visit_history)
    if history_count >= 4:
        confidence_label = "HIGH"
        confidence_note  = f"Based on {history_count} visits of history"
    elif history_count >= 2:
        confidence_label = "MODERATE"
        confidence_note  = f"Based on {history_count} visits of history"
    elif history_count == 1:
        confidence_label = "LOW"
        confidence_note  = "Only 1 previous visit available"
    else:
        confidence_label = "FIRST VISIT"
        confidence_note  = "No prior history — rules only"

    return {
        "overall_risk":     rules_result["overall_risk"],
        "risk_color":       rules_result["risk_color"],
        "matri_score":      final_score,
        "confidence_label": confidence_label,
        "confidence_note":  confidence_note,
        "domain_scores":    rules_result["domain_scores"],
        "trend_flags":      trend_result["trend_flags"],
        "trend_details":    trend_result["trend_details"],
        "flagged_values":   rules_result["flagged_values"],
        "explanations":     rules_result["explanations"],
    }