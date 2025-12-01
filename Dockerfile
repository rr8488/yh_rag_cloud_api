# Use a robust Python base image
FROM python:3.11-slim

# Set working directory inside the container
WORKDIR /app

# 1. Install necessary system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        gcc \
        build-essential \
        libpq-dev \
        pandoc \
        tesseract-ocr && \
    rm -rf /var/lib/apt/lists/*

# 2. Copy dependency file
COPY requirements.txt /app/requirements.txt

# 3. Install Python dependencies
# FORCE UPGRADE of google-generativeai to the latest version
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir --upgrade "google-generativeai>=0.8.3"

# 4. Copy the rest of the application code
COPY . /app

# 5. Define the port
ENV PORT 8080

# 6. Run the application
CMD exec uvicorn yh_rag_cloud_api.app:app --host 0.0.0.0 --port ${PORT}