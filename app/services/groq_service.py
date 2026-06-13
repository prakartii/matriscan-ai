# app/services/groq_service.py
import json
import logging
from groq import Groq
from app.config import GROQ_API_KEY

logger = logging.getLogger("matriscan")
client = Groq(api_key=GROQ_API_KEY)

# ── Supported languages for Whisper ─────────────────────────────────
WHISPER_LANGUAGE_CODES = {
    "ta": "ta",   # Tamil
    "hi": "hi",   # Hindi
    "te": "te",   # Telugu
    "kn": "kn",   # Kannada
    "ml": "ml",   # Malayalam
    "bn": "bn",   # Bengali
    "mr": "mr",   # Marathi
    "gu": "gu",   # Gujarati
    "or": "or",   # Odia
    "pa": "pa",   # Punjabi
}


# ── STT: Whisper Large V3 ────────────────────────────────────────────
async def transcribe_audio(audio_bytes: bytes, filename: str, language: str = "ta") -> dict:
    """
    Send audio to Groq Whisper Large V3.
    Whisper auto-detects language — the language param is a hint
    that improves accuracy for regional Indian languages.
    Returns transcript text.
    """
    try:
        whisper_lang = WHISPER_LANGUAGE_CODES.get(language, None)

        # Groq expects a file-like tuple: (filename, bytes, mimetype)
        response = client.audio.transcriptions.create(
            model="whisper-large-v3",
            file=(filename, audio_bytes, "audio/webm"),
            language=whisper_lang,
            response_format="text"
        )

        transcript = response if isinstance(response, str) else response.text
        logger.info(f"Whisper transcript ({language}): {transcript[:80]}...")
        return {"success": True, "transcript": transcript}

    except Exception as e:
        logger.error(f"Whisper error: {e}")
        return {"success": False, "transcript": "", "error": str(e)}


# ── NLP: Symptom extraction via Llama 3.1 ───────────────────────────
def extract_symptoms(transcript: str, language: str = "ta") -> dict:
    """
    Send transcript to Llama 3.1 to extract structured symptoms.
    Always returns symptoms in English internally regardless of
    input language — the rules engine only understands English keys.
    """
    try:
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a clinical NLP system for maternal health in India. "
                        "Extract symptoms from transcripts spoken by rural health workers "
                        "in any Indian language. "
                        "ALWAYS return symptoms in English regardless of input language. "
                        "Return ONLY valid JSON, no explanation, no markdown."
                    )
                },
                {
                    "role": "user",
                    "content": f"""Extract symptoms from this transcript.
Transcript: {transcript}

Return ONLY this JSON structure:
{{
  "symptoms": ["headache", "blurred_vision"],
  "severity": {{"headache": "severe", "blurred_vision": "moderate"}},
  "duration_hints": ["since two days"],
  "language_detected": "Tamil"
}}

Valid symptom keys: headache, blurred_vision, epigastric_pain, fever,
reduced_fetal_movement, vaginal_bleeding, swelling, dysuria, foul_discharge"""
                }
            ],
            temperature=0.1,   # low temperature = consistent structured output
            max_tokens=300
        )

        raw = response.choices[0].message.content.strip()
        # Strip markdown fences if Llama adds them
        raw = raw.replace("```json", "").replace("```", "").strip()
        parsed = json.loads(raw)
        logger.info(f"Symptoms extracted: {parsed.get('symptoms', [])}")
        return {"success": True, **parsed}

    except json.JSONDecodeError as e:
        logger.error(f"Llama JSON parse error: {e}, raw: {raw}")
        return {"success": False, "symptoms": [], "severity": {}, "duration_hints": []}
    except Exception as e:
        logger.error(f"Llama symptom extraction error: {e}")
        return {"success": False, "symptoms": [], "severity": {}, "duration_hints": []}


# ── Care plan generation via Llama 3.1 ──────────────────────────────
def generate_care_plan(patient: dict, risk_result: dict, language: str = "ta") -> dict:
    """
    Generate structured care plan using Llama 3.1.
    doctor_advice is always in English (for the doctor).
    local_counseling is in the patient's language (for the ANM to read aloud).
    """

    # Map language code to full name for the prompt
    language_names = {
        "ta": "Tamil", "hi": "Hindi", "te": "Telugu",
        "kn": "Kannada", "ml": "Malayalam", "bn": "Bengali",
        "mr": "Marathi", "gu": "Gujarati", "or": "Odia", "pa": "Punjabi"
    }
    language_name = language_names.get(language, "Tamil")

    try:
        flagged = ", ".join(risk_result.get("flagged_values", []))
        domains = risk_result.get("domain_scores", {})
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a maternal health clinical decision support system "
                        "trained on WHO ANC guidelines and Government of India JSSK protocol. "
                        "Generate structured care plans for rural ANM health workers in India. "
                        "Return ONLY valid JSON, no explanation, no markdown."
                    )
                },
                {
                    "role": "user",
                    "content": f"""Generate a care plan for this patient.

Patient: Age {patient.get('age')}, {patient.get('parity')}, gestational age {patient.get('gestational_age', 'unknown')} weeks
Risk level: {risk_result.get('overall_risk')}
MatriScore: {risk_result.get('matri_score')}
Flagged values: {flagged}
Domain scores: {domains}

Return ONLY this JSON:
{{
  "doctor_advice": "Clinical advice in English for the doctor",
  "local_counseling": "Simple counseling in {language_name} language for the patient",
  "followup_days": 7,
  "referral_urgency": "within 24 hours",
  "red_flags": ["symptom to watch for"]
}}

Rules:
- doctor_advice: always English, clinical terminology, WHO/JSSK protocol
- local_counseling: always in {language_name}, simple words a village woman understands
- followup_days: 0 if RED (immediate), 3-7 if AMBER, 14-28 if GREEN
- referral_urgency: "immediate" / "within 6 hours" / "within 24 hours" / "routine"
- red_flags: 2-4 warning signs to watch for"""
                }
            ],
            temperature=0.3,
            max_tokens=600
        )

        raw = response.choices[0].message.content.strip()
        raw = raw.replace("```json", "").replace("```", "").strip()
        parsed = json.loads(raw)
        logger.info(f"Care plan generated for risk: {risk_result.get('overall_risk')}")
        return {"success": True, **parsed}

    except json.JSONDecodeError as e:
        logger.error(f"Care plan JSON parse error: {e}")
        return _fallback_care_plan(language_name, risk_result)
    except Exception as e:
        logger.error(f"Care plan generation error: {e}")
        return _fallback_care_plan(language_name, risk_result)


def _fallback_care_plan(language_name: str, risk_result: dict) -> dict:
    """
    If Llama fails for any reason, return a safe generic plan.
    Never let a care plan endpoint return empty — that's a patient safety issue.
    """
    return {
        "success": False,
        "doctor_advice": (
            f"Risk level: {risk_result.get('overall_risk')}. "
            "Follow standard WHO ANC protocol. "
            "Refer to nearest FRU if HIGH risk."
        ),
        "local_counseling": f"Please consult your doctor immediately.",
        "followup_days": 0 if risk_result.get("overall_risk") == "HIGH" else 7,
        "referral_urgency": "immediate" if risk_result.get("risk_color") == "RED" else "routine",
        "red_flags": ["high blood pressure", "reduced fetal movement", "heavy bleeding"]
    }
# ── OCR: lab report image -> structured values (Llama 4 Scout vision) ──
import base64

VISION_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"

def extract_lab_values(image_bytes: bytes, mime_type: str = "image/jpeg") -> dict:
    """
    Send a printed lab-report photo to Groq's vision model and get back
    structured lab values as JSON. One call, no PaddleOCR. (Online path.)
    """
    b64 = base64.b64encode(image_bytes).decode("utf-8")

    try:
        response = client.chat.completions.create(
            model=VISION_MODEL,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": (
                                "This is a printed Indian medical lab report. "
                                "Extract the lab values. Return ONLY valid JSON, "
                                "no markdown, no explanation, with this exact shape:\n"
                                "{\n"
                                '  "hb": {"value": null, "unit": "g/dL"},\n'
                                '  "bp_systolic": {"value": null},\n'
                                '  "bp_diastolic": {"value": null},\n'
                                '  "glucose_fasting": {"value": null, "unit": "mg/dL"},\n'
                                '  "glucose_pp": {"value": null, "unit": "mg/dL"},\n'
                                '  "urine_protein": {"value": null},\n'
                                '  "platelets": {"value": null}\n'
                                "}\n"
                                "Use null for any value not present. "
                                "For urine_protein use strings like \"2+\" or \"negative\"."
                            ),
                        },
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:{mime_type};base64,{b64}"},
                        },
                    ],
                }
            ],
            temperature=0.1,
            max_tokens=500,
            response_format={"type": "json_object"},
        )

        raw = response.choices[0].message.content.strip()
        raw = raw.replace("```json", "").replace("```", "").strip()
        parsed = json.loads(raw)
        logger.info(f"Groq vision lab values: {list(parsed.keys())}")
        return {"success": True, "values": parsed}

    except json.JSONDecodeError as e:
        logger.error(f"Groq vision JSON parse error: {e}")
        return {"success": False, "values": {}, "error": "parse_error"}
    except Exception as e:
        logger.error(f"Groq vision OCR error: {e}")
        return {"success": False, "values": {}, "error": str(e)}