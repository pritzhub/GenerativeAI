import os
from dotenv import load_dotenv
from openai import OpenAI

# Load .env in dev; in production it will simply do nothing if .env is absent
load_dotenv()

def get_openai_client() -> OpenAI:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set.")
    return OpenAI(api_key=api_key)
