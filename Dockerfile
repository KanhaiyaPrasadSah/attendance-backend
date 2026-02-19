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

# Upgrade pip
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip

# Install numpy separately first (helps prevent memory spikes)
RUN pip install --no-cache-dir numpy==1.23.5

# Install the rest of the requirements
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . .

# NOTE: We removed the pre-download step to keep build memory under 512MB.
# The models will download automatically when the service first starts.

# Start the FastAPI application
CMD ["uvicorn main:app --host 0.0.0.0 --port 10000"]
