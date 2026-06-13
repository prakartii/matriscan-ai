# app/services/sarvam_service.py
import logging
import httpx
from app.config import SARVAM_API_KEY

logger = logging.getLogger("matriscan")

SARVAM_TTS_URL = "https://api.sarvam.ai/text-to-speech"

# Your app uses short codes ("ta"); Sarvam wants BCP-47 ("ta-IN").
LANG_TO_BCP47 = {
    "ta": "ta-IN", "hi": "hi-IN", "te": "te-IN", "kn": "kn-IN",
    "ml": "ml-IN", "bn": "bn-IN", "mr": "mr-IN", "gu": "gu-IN",
    "or": "od-IN", "pa": "pa-IN", "en": "en-IN",
}


def synthesize_speech(text: str, language: str = "ta") -> dict:
    """
    Call Sarvam Bulbul v3 TTS. Returns base64-encoded WAV audio.
    Accepts either a short code ('ta') or a full BCP-47 code ('ta-IN').
    """
    # Normalise: if they already passed 'ta-IN', keep it; else map it.
    target = language if "-" in language else LANG_TO_BCP47.get(language, "ta-IN")

    payload = {
        "text": text[:2500],            # v3 hard limit is 2500 chars
        "target_language_code": target,
        "model": "bulbul:v3",
        "speaker": "ritu",              # female voice; see docs for full list
        "pace": 0.9,                    # slightly slower for clarity
    }
    headers = {
        "api-subscription-key": SARVAM_API_KEY,
        "Content-Type": "application/json",
    }

    try:
        resp = httpx.post(SARVAM_TTS_URL, json=payload, headers=headers, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        audios = data.get("audios", [])
        if not audios:
            logger.error(f"Sarvam returned no audio: {data}")
            return {"success": False, "audio_base64": "", "error": "no audio returned"}

        return {
            "success": True,
            "audio_base64": audios[0],       # base64 WAV string
            "mime_type": "audio/wav",
            "language": target,
        }

    except httpx.HTTPStatusError as e:
        logger.error(f"Sarvam HTTP {e.response.status_code}: {e.response.text}")
        return {"success": False, "audio_base64": "", "error": f"http {e.response.status_code}"}
    except Exception as e:
        logger.error(f"Sarvam TTS error: {e}")
        return {"success": False, "audio_base64": "", "error": str(e)}