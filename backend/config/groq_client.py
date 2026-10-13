
import os
from pathlib import Path
from dotenv import load_dotenv
from groq import Groq

# Load .env from the backend folder
BACKEND_DIR = Path(__file__).resolve().parents[1]
ENV_PATH = BACKEND_DIR / ".env"

load_dotenv(ENV_PATH, override=True)

api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    raise ValueError(f"GROQ_API_KEY not found. Check: {ENV_PATH}")

client = Groq(api_key=api_key)
MODEL_NAME = "openai/gpt-oss-120b"
