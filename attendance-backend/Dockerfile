FROM python:3.9-slim

RUN apt-get update && apt-get install -y libgl1-mesa-glx libglib2.0-0
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .

# Pre-download models so the first request isn't slow
RUN python -c "from insightface.app import FaceAnalysis; FaceAnalysis(name='buffalo_l').prepare(ctx_id=-1)"

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "10000"]