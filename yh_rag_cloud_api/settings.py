from typing import Literal, List, Any, Optional
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# ---
# NOTE: We have REMOVED the manual load_dotenv() logic from this file.
# It is now handled by uvicorn in `run_dev.py`.
# ---

class Settings(BaseSettings):
    """Centralized environment configuration for yh_rag_cloud_api."""

    # --- Core environment ---
    env: Literal["dev", "prod", "staging", "test", "local"] = Field("dev", alias="ENV")
    app_host: str = Field("127.0.0.1", alias="APP_HOST")
    app_port: int = Field(8088, alias="APP_PORT")
    ollama_host: str = Field("http://127.0.0.1:11434", alias="OLLAMA_HOST")

    # --- Database ---
    database_url: str = Field(..., alias="DATABASE_URL") 

    # --- Google Cloud & Firebase ---
    gemini_api_key: str = Field(..., alias="GEMINI_API_KEY")
    gemini_model_name: str = Field(..., alias="GEMINI_MODEL_NAME")
    firebase_project_id: str = Field(..., alias="FIREBASE_PROJECT_ID")
    firebase_database_url: str = Field(..., alias="FIREBASE_DATABASE_URL")
    google_cloud_project: str = Field(..., alias="GOOGLE_CLOUD_PROJECT") 
    google_application_credentials: Optional[str] = Field(None, alias="GOOGLE_APPLICATION_CREDENTIALS")

    # --- Document AI & GCS ---
    document_ai_processor_id: str = Field("70b55623f60923", alias="DOCUMENT_AI_PROCESSOR_ID") 
    document_ai_location: str = Field("us", alias="DOCUMENT_AI_LOCATION") 
    gcs_bucket_name: str = Field("fluyx-yh-kb.firebasestorage.app", alias="GCS_BUCKET_NAME")

    # --- OCR configuration ---
    ocr_fallback: bool = Field(True, alias="OCR_FALLBACK")
    ocr_engine: Literal["tesseract", "easyocr", "paddle", "google_vision"] = Field("tesseract", alias="OCR_ENGINE")
    ocr_dpi: int = Field(300, alias="OCR_DPI")
    ocr_rotate: bool = Field(True, alias="OCR_ROTATE")
    ocr_trim: bool = Field(True, alias="OCR_TRIM")
    ocr_langs: List[str] = Field(default_factory=lambda: ["eng"], alias="OCR_LANGS")

    @field_validator("ocr_langs", mode="before")
    @classmethod
    def _split_langs(cls, v: Any) -> List[str]:
        if isinstance(v, str):
            return [s.strip() for s in v.split(",") if s.strip()]
        return v

    # --- Model config ---
    model_config = SettingsConfigDict(
        # Let uvicorn handle loading the .env file. Pydantic will just read the environment variables.
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

# This line will now run *inside* the child process
# which has the correct environment variables.
settings = Settings()
print("--- Settings loaded successfully. ---")