# app/services/risk.py
from typing import List, Dict, Any
from app.engine.rules import run_rules_engine, SEVERITY_SEVERE, SEVERITY_MODERATE
from app.schemas.models import VisitRecord, PatientInfo

BP_RISING_SLOPE_THRESHOLD  =  2.0
BP_FALLING_SLOPE_THRESHOLD = -2.0
HB_FALLING_SLOPE_THRESHOLD = -0.3


def _slope(xs: List[float], ys: List[float]) -> float:
    """Least-squares linear regression slope (pure Python)."""
    n = len(xs)
    mean_x = sum(xs) / n
    mean_y = sum(ys) / n
    num = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    den = sum((x - mean_x) ** 2 for x in xs)
    return num / den if den != 0 else 0.0


def _compute_trends(visit_history: List[VisitRecord]) -> Dict[str, Any]:
    if len(visit_history) < 2:
        return {"trend_flags": [], "trend_details": {}, "score_boost": 0}

    records = sorted(
        [
            {
                "visit_number": v.visit_number,
                "bp_systolic":  v.vitals.bp_systolic,
                "hemoglobin":   v.vitals.hemoglobin,
            }
            for v in visit_history
        ],
        key=lambda r: r["visit_number"],
    )

    xs = [float(r["visit_number"]) for r in records]
    trend_flags: List[str] = []
    trend_details: Dict[str, float] = {}
    score_boost = 0

    # BP systolic trend
    bp_pairs = [(x, r["bp_systolic"]) for x, r in zip(xs, records) if r["bp_systolic"] is not None]
    if len(bp_pairs) >= 2:
        bx, by = zip(*bp_pairs)
        slope = _slope(list(bx), list(by))
        trend_details["bp_systolic_slope"] = round(slope, 2)
        if slope > BP_RISING_SLOPE_THRESHOLD:
            trend_flags.append("BP_RISING")
            score_boost += 8
        elif slope < BP_FALLING_SLOPE_THRESHOLD:
            trend_flags.append("BP_FALLING")

    # Hemoglobin trend
    hb_pairs = [(x, r["hemoglobin"]) for x, r in zip(xs, records) if r["hemoglobin"] is not None]
    if len(hb_pairs) >= 2:
        hx, hy = zip(*hb_pairs)
        slope = _slope(list(hx), list(hy))
        trend_details["hb_slope"] = round(slope, 2)
        if slope < HB_FALLING_SLOPE_THRESHOLD:
            trend_flags.append("HB_FALLING")
            score_boost += 6

    return {
        "trend_flags":   trend_flags,
        "trend_details": trend_details,
        "score_boost":   score_boost,
    }


def run_full_analysis(
    patient: PatientInfo,
    current_visit: VisitRecord,
    visit_history: List[VisitRecord],
) -> Dict[str, Any]:
    rules_result = run_rules_engine(current_visit.vitals, current_visit.symptoms)

    full_history = visit_history + [current_visit]
    trend_result = _compute_trends(full_history)

    final_score = min(rules_result["matri_score"] + trend_result["score_boost"], 100)

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
