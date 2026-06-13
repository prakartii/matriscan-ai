# app/services/paddle_service.py
import re
import logging
from paddleocr import PaddleOCR

logger = logging.getLogger("matriscan")

# Load PaddleOCR ONCE at module import, not per request (it's heavy).
# Models are already cached locally, so this runs fully offline.
_ocr_engine = None

def _get_engine():
    global _ocr_engine
    if _ocr_engine is None:
        logger.info("Loading PaddleOCR engine (one-time)...")
        _ocr_engine = PaddleOCR(use_textline_orientation=True, lang="en")
    return _ocr_engine


def _extract_raw_text(image_path: str) -> str:
    """Run PaddleOCR and flatten all detected text into one string."""
    engine = _get_engine()
    result = engine.predict(image_path)

    lines = []
    # PaddleOCR 3.x returns a list of result objects; text is under 'rec_texts'
    for page in result:
        texts = page.get("rec_texts", []) if isinstance(page, dict) else []
        lines.extend(texts)
    return "\n".join(lines)


# ── Local parsing: raw OCR text -> structured values (NO LLM, fully offline) ──
#
# Each pattern looks for a label then captures the number near it.
# Tune these once you see how YOUR lab report actually prints each field.

def _find_number(text: str, patterns: list[str]) -> float | None:
    for pat in patterns:
        m = re.search(pat, text, re.IGNORECASE)
        if m:
            try:
                return float(m.group(1))
            except (ValueError, IndexError):
                continue
    return None


def parse_lab_values(raw_text: str) -> dict:
    """
    Tolerant parser for common Indian antenatal lab-report formats.
    Handles label variants, ':' or '=' or whitespace separators, and
    values appearing on the same line as the label.
    NOTE: still won't catch every layout — manual override covers the rest.
    """
    # Normalise whitespace so patterns are simpler.
    t = re.sub(r"[ \t]+", " ", raw_text)

    hb = _find_number(t, [
        r"h(?:a?emoglobin|gb)\s*[:=\-]?\s*(\d{1,2}(?:\.\d)?)",
        r"\bhb\b\s*[:=\-]?\s*(\d{1,2}(?:\.\d)?)",
        r"\bhb\b\D{0,10}(\d{1,2}\.\d)",          # Hb followed by a decimal nearby
    ])

    # Blood pressure: "148/96", "BP 148 / 96", "B.P : 148-96"
    bp_sys = bp_dia = None
    bp_match = re.search(
        r"b\.?\s*p\.?\s*[:=\-]?\s*(\d{2,3})\s*[/\-]\s*(\d{2,3})", t, re.IGNORECASE)
    if not bp_match:  # fallback: any "NNN/NN" that looks like BP
        bp_match = re.search(r"\b(1[0-2]\d|[7-9]\d)\s*/\s*([4-9]\d|1[0-2]\d)\b", t)
    if bp_match:
        bp_sys, bp_dia = float(bp_match.group(1)), float(bp_match.group(2))

    glucose_fasting = _find_number(t, [
        r"fasting\s*(?:blood\s*)?(?:sugar|glucose)?\s*[:=\-]?\s*(\d{2,3})",
        r"\bfbs\b\s*[:=\-]?\s*(\d{2,3})",
        r"\bf\.?b\.?s\.?\b\D{0,8}(\d{2,3})",
    ])
    glucose_pp = _find_number(t, [
        r"post[\s\-]*prandial\s*(?:sugar|glucose)?\s*[:=\-]?\s*(\d{2,3})",
        r"\bpp(?:bs)?\b\s*[:=\-]?\s*(\d{2,3})",
        r"2\s*hr?\s*(?:pp)?\D{0,8}(\d{2,3})",
    ])

    protein = None
    pm = re.search(
        r"(?:urine\s*)?(?:protein|albumin)\s*[:=\-]?\s*"
        r"(nil|trace|negative|absent|present|\d\s*\+|\+{1,4})",
        t, re.IGNORECASE)
    if pm:
        protein = re.sub(r"\s+", "", pm.group(1))

    platelets = _find_number(t, [
        r"platelet[s]?\s*(?:count)?\s*[:=\-]?\s*(\d{1,3}(?:\.\d)?)",
        r"\bplt\b\D{0,8}(\d{1,3}(?:\.\d)?)",
    ])

    return {
        "hb":              {"value": hb,              "unit": "g/dL"},
        "bp_systolic":     {"value": bp_sys},
        "bp_diastolic":    {"value": bp_dia},
        "glucose_fasting": {"value": glucose_fasting, "unit": "mg/dL"},
        "glucose_pp":      {"value": glucose_pp,      "unit": "mg/dL"},
        "urine_protein":   {"value": protein},
        "platelets":       {"value": platelets},
    }


def extract_lab_values_offline(image_path: str) -> dict:
    """Full offline pipeline: PaddleOCR text extraction -> local parsing."""
    try:
        raw = _extract_raw_text(image_path)
        if not raw.strip():
            logger.warning("PaddleOCR found no text in image")
            return {"success": False, "values": {}, "raw_text": "", "error": "no text"}

        values = parse_lab_values(raw)
        logger.info(f"Offline OCR parsed: { {k: v.get('value') for k, v in values.items()} }")
        return {"success": True, "values": values, "raw_text": raw}

    except Exception as e:
        logger.error(f"Offline OCR error: {e}")
        return {"success": False, "values": {}, "raw_text": "", "error": str(e)}