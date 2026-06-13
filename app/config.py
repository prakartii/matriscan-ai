# app/config.py
from dotenv import load_dotenv
import os

load_dotenv()

# API Keys
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GOOGLE_APPLICATION_CREDENTIALS = os.getenv("GOOGLE_APPLICATION_CREDENTIALS", "")
SARVAM_API_KEY = os.getenv("SARVAM_API_KEY", "")

# Mock mode - when True, no real API calls are made
# This is your demo safety net
MOCK_MODE = os.getenv("MOCK_MODE", "true").lower() == "true"

# App settings
APP_PORT = int(os.getenv("APP_PORT", "8001"))
ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "http://localhost:3000").split(",")