from mistralai import Mistral
from dotenv import load_dotenv
import os

load_dotenv()

client = Mistral(
    api_key=os.getenv("MISTRAL_API_KEY")
)