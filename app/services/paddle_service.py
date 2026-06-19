# app/services/paddle_service.py
import re
import logging

logger = logging.getLogger("matriscan")

_ocr_engine = None


def _get_engine():
    global _ocr_engine
    if _ocr_engine is None:
        try:
            from paddleocr import PaddleOCR  # optional; not installed on Render
            logger.info("Loading PaddleOCR engine (one-time)...")
            _ocr_engine = PaddleOCR(use_textline_orientation=True, lang="en")
        except ImportError:
            logger.warning("PaddleOCR not installed — offline OCR unavailable")
            _ocr_engine = None
    return _ocr_engine


def _extract_raw_text(image_path: str) -> str:
    engine = _get_engine()
    if engine is None:
        raise RuntimeError("PaddleOCR is not installed")
    result = engine.predict(image_path)
    lines = []
    for page in result:
        texts = page.get("rec_texts", []) if isinstance(page, dict) else []
        lines.extend(texts)
    return "\n".join(lines)


def _find_number(text: str, patterns: list) -> float | None:
    for pat in patterns:
        m = re.search(pat, text, re.IGNORECASE)
        if m:
            try:
                return float(m.group(1))
            except (ValueError, IndexError):
                continue
    return None


def parse_lab_values(raw_text: str) -> dict:
    t = re.sub(r"[ \t]+", " ", raw_text)

    hb = _find_number(t, [
        r"h(?:a?emoglobin|gb)\s*[:=\-]?\s*(\d{1,2}(?:\.\d)?)",
        r"\bhb\b\s*[:=\-]?\s*(\d{1,2}(?:\.\d)?)",
        r"\bhb\b\D{0,10}(\d{1,2}\.\d)",
    ])

    bp_sys = bp_dia = None
    bp_match = re.search(
        r"b\.?\s*p\.?\s*[:=\-]?\s*(\d{2,3})\s*[/\-]\s*(\d{2,3})", t, re.IGNORECASE)
    if not bp_match:
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
