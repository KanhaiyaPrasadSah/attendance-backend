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
# We install one by one to keep memory spikes low
RUN pip install --no-cache-dir numpy==1.23.5
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . .

# We REMOVED the "RUN python -c..." line to save memory during build

# Start the FastAPI application
# The models will download the very first time you visit the URL
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "10000"]
