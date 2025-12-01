import os
import uvicorn
from dotenv import load_dotenv  # 1. Import load_dotenv

load_dotenv()  # 2. Call load_dotenv() to read .env file

HOST = os.getenv("APP_HOST", "127.0.0.1")
PORT = int(os.getenv("APP_PORT", "8088"))

APP_IMPORT = "yh_rag_cloud_api.app:app"
RELOAD_DIRS = ["yh_rag_cloud_api"]
RELOAD_EXCLUDES = [".venv", ".git", "__pycache__", "functions"]


if __name__ == "__main__":
    print(f"🚀 Starting FastAPI dev server on http://{HOST}:{PORT}")
    uvicorn.run(
        APP_IMPORT,
        host=HOST,
        port=PORT,
        reload=True,
        reload_excludes=RELOAD_EXCLUDES,
        reload_dirs=RELOAD_DIRS,
        # Optional: Add env_file to ensure Uvicorn's reloader also reads it,
        # though load_dotenv() usually suffices before calling uvicorn.run().
        # env_file=".env"
    )