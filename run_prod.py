import os
import shlex
import subprocess

HOST = os.getenv("APP_HOST", "0.0.0.0")
PORT = int(os.getenv("APP_PORT", "8080")) # Default to 8080, a common prod port

APP_IMPORT = "yh_rag_cloud_api.app:app"

# Use Gunicorn for production: https://gunicorn.org/
# It's a battle-tested WSGI server. Uvicorn provides a Gunicorn-compatible worker class.
WORKER_CLASS = "uvicorn.workers.UvicornWorker"
WORKERS = int(os.getenv("WEB_CONCURRENCY", "2")) # Number of worker processes
THREADS = int(os.getenv("PYTHON_MAX_THREADS", "4")) # Number of threads per worker

if __name__ == "__main__":
    print(f"🚀 Starting FastAPI production server on http://{HOST}:{PORT}")

    # Construct the gunicorn command
    gunicorn_cmd = (
        f"gunicorn {APP_IMPORT} --bind {HOST}:{PORT} "
        f"--workers {WORKERS} --threads {THREADS} --worker-class {WORKER_CLASS}"
    )

    # Use subprocess.run to execute the command
    subprocess.run(shlex.split(gunicorn_cmd))