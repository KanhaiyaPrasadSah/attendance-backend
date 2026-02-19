# Use a stable Python base
FROM python:3.9-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    build-essential \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Force low memory usage for AI libraries
ENV INFERENCE_MODE=true
ENV ONNXRUNTIME_EXECUTION_MODE=SEQUENTIAL
ENV OMP_NUM_THREADS=1

WORKDIR /app

# Install requirements in small batches to save RAM
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir numpy==1.23.5
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Correct Exec-form command for Render to avoid exit 128
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "10000", "--workers", "1"]
