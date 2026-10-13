
import os
from pathlib import Path
from dotenv import load_dotenv
from groq import Groq

# Load .env from project root
env_path = Path(__file__).resolve().parents[2] / ".env"
load_dotenv(env_path)

api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    raise ValueError("GROQ_API_KEY is missing from the root .env file")

client = Groq(api_key=api_key)

MODEL_NAME = "openai/gpt-oss-120b"