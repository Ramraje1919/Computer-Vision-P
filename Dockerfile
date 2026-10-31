# Dockerfile for MOT17 Multi-Object Tracking Pipeline
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies for OpenCV and multimedia processing
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    ffmpeg \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install python dependencies
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

# Copy source tree and configurations
COPY config/ /app/config/
COPY src/ /app/src/
COPY scripts/ /app/scripts/
COPY weights/ /app/weights/
COPY main.py /app/

ENV PYTHONPATH=/app

CMD ["python", "main.py", "--help"]

