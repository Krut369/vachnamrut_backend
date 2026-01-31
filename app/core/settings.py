import os
from dotenv import load_dotenv
from typing import List

load_dotenv()

class Settings:
    PROJECT_NAME: str = "Vachanamrut AI"
    VERSION: str = "1.0.0"
    
    # API Keys - Support Comma Separated for Rotation
    _groq_keys_str: str = os.getenv("GROQ_API_KEY", "")
    GROQ_API_KEYS: List[str] = [k.strip() for k in _groq_keys_str.split(",") if k.strip()]

    _gemini_keys_str: str = os.getenv("GEMINI_API_KEYS", "")
    GEMINI_API_KEYS: List[str] = [k.strip() for k in _gemini_keys_str.split(",") if k.strip()]
    
    # Paths
    VECTOR_DB_PATH: str = "./data/vachanamrut_db"
    JSON_DATA_PATH: str = "./data/vachanamrut_cleaned.json"
    
    # Model Config
    LLM_MODEL: str = "llama-3.3-70b-versatile"
    EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"

settings = Settings()
