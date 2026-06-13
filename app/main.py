# app/main.py
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import logging
from app.services.sarvam_service import synthesize_speech
from app.services.paddle_service import extract_lab_values_offline
import tempfile, os
from app.config import MOCK_MODE, ALLOWED_ORIGINS
from app.schemas.models import AnalyzeRequest, CarePlanRequest, TTSRequest
from app.services.risk import run_full_analysis
from app.services.groq_service import (
    transcribe_audio, extract_symptoms, generate_care_plan, extract_lab_values
)
from app.mock.responses import (
    get_mock_careplan_response, get_mock_transcribe_response,
    get_mock_ocr_response, get_mock_tts_response
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("matriscan")

app = FastAPI(
    title="MatriScan AI Service",
    description="Maternal health risk assessment API",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Health check ─────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {
        "status": "ok",
        "mock_mode": MOCK_MODE,
        "service": "MatriScan AI"
    }


# ── Risk analysis — always real, never mocked ────────────────────────
@app.post("/ai/analyze")
def analyze_risk(request: AnalyzeRequest):
    logger.info(f"Analyzing risk for patient {request.patient.patient_id}")
    return run_full_analysis(
        patient=request.patient,
        current_visit=request.current_visit,
        visit_history=request.visit_history
    )


# ── Voice transcription — real Groq Whisper ──────────────────────────
@app.post("/ai/transcribe")
async def transcribe_audio_endpoint(
    audio: UploadFile = File(...),
    language: str = Form(default="ta")
):
    if MOCK_MODE:
        return get_mock_transcribe_response(language)

    try:
        audio_bytes = await audio.read()

        # Step 1: Whisper STT
        stt_result = await transcribe_audio(audio_bytes, audio.filename or "audio.webm", language)

        if not stt_result["success"] or not stt_result["transcript"]:
            logger.warning("Whisper failed, falling back to mock")
            return get_mock_transcribe_response(language)

        # Step 2: Llama symptom extraction
        symptom_result = extract_symptoms(stt_result["transcript"], language)

        return {
            "transcript": stt_result["transcript"],
            "language_detected": symptom_result.get("language_detected", language),
            "symptoms": symptom_result.get("symptoms", []),
            "severity": symptom_result.get("severity", {}),
            "duration_hints": symptom_result.get("duration_hints", [])
        }

    except Exception as e:
        logger.error(f"Transcribe endpoint error: {e}")
        return get_mock_transcribe_response(language)


# ── Care plan — real Groq Llama ───────────────────────────────────────
@app.post("/ai/careplan")
def generate_careplan(request: CarePlanRequest):
    if MOCK_MODE:
        return get_mock_careplan_response(request.language)

    try:
        patient_dict = {
            "age": request.patient.age,
            "parity": request.patient.parity,
            "gestational_age": request.risk_result.get("gestational_age", "unknown")
        }

        result = generate_care_plan(patient_dict, request.risk_result, request.language)

        if not result.get("success"):
            logger.warning("Care plan generation failed, using fallback")

        return result

    except Exception as e:
        logger.error(f"Careplan endpoint error: {e}")
        return get_mock_careplan_response(request.language)


# ── OCR — PaddleOCR comes in Phase 2b ────────────────────────────────
@app.post("/ai/ocr")
async def ocr_lab_report(image: UploadFile = File(...), offline: bool = False):
    if MOCK_MODE:
        return get_mock_ocr_response()

    image_bytes = await image.read()

    # Offline path: save to temp file, run PaddleOCR locally
    if offline:
        suffix = os.path.splitext(image.filename or "img.jpg")[1] or ".jpg"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(image_bytes)
            tmp_path = tmp.name
        try:
            result = extract_lab_values_offline(tmp_path)
        finally:
            os.unlink(tmp_path)
        if not result.get("success"):
            return get_mock_ocr_response()
        return result["values"]

    # Online path: Groq vision (already built)
    mime = image.content_type or "image/jpeg"
    result = extract_lab_values(image_bytes, mime)
    if not result.get("success"):
        logger.warning("Groq vision OCR failed, using mock")
        return get_mock_ocr_response()
    return result["values"]

# ── TTS — Sarvam TTS comes in Phase 3 ──────────────────────────
@app.post("/ai/tts")
def text_to_speech(request: TTSRequest):
    if MOCK_MODE:
        return get_mock_tts_response(request.language)

    result = synthesize_speech(request.text, request.language)
    if not result.get("success"):
        logger.warning("Sarvam TTS failed, using mock")
        return get_mock_tts_response(request.language)
    return result