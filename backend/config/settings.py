"""Application settings and environment configuration."""

from dataclasses import dataclass
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


@dataclass
class Settings:
    app_name: str = "Recommendation Backend"
    env: str = os.getenv("ENV", "development")
    data_dir: Path = Path(os.getenv("DATA_DIR", Path(__file__).resolve().parents[1] / "data"))
    chroma_dir: Path = Path(os.getenv("CHROMA_DIR", Path(__file__).resolve().parents[1] / "rag" / "chroma_db"))
    openai_api_key: str | None = os.getenv("OPENAI_API_KEY")


settings = Settings()
