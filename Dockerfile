 # Use a stable Python base
FROM python:3.9-slim

# Install system dependencies and C++ compiler (g++)
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglx-mesa0 \
    libglib2.0-0 \
    build-essential \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory
WORKDIR /app

# Upgrade pip and install requirements
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . .

# Pre-download the AI models (Buffalo_L) 
# This prevents the server from crashing due to long download times on first request
RUN python -c "from insightface.app import FaceAnalysis; face_app = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider']); face_app.prepare(ctx_id=-1)"

# Start the FastAPI application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "10000"]
