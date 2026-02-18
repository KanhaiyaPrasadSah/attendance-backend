import numpy as np
import faiss
import requests
from fastapi import FastAPI, UploadFile, File, Form
from insightface.app import FaceAnalysis
import cv2

app = FastAPI()

# 1. Initialize AI Model (Buffalo_L for High Accuracy)
face_app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
face_app.prepare(ctx_id=-1, det_size=(640, 640))

# 2. In-Memory Database (FAISS)
index = faiss.IndexFlatIP(512) # FlatIP is used for Cosine Similarity
index_keys = [] # To store Hall Ticket Numbers

APPSCRIPT_URL = "YOUR_APPSCRIPT_URL"

@app.on_event("startup")
async def load_database():
    """Pulls existing vectors from Google Sheets into RAM on startup"""
    global index_keys
    try:
        response = requests.get(f"{APPSCRIPT_URL}?action=getVectors").json()
        if response["status"] == "success":
            embeddings = []
            for item in response["data"]:
                # Convert string vector back to numpy array
                vec = np.fromstring(item["vector"], sep=',').astype('float32')
                embeddings.append(vec)
                index_keys.append(item["htNo"])
            
            if embeddings:
                index.add(np.stack(embeddings))
                print(f"✅ Loaded {len(index_keys)} students into RAM.")
    except Exception as e:
        print(f"❌ Startup Sync Failed: {e}")

@app.post("/identify")
async def identify(file: UploadFile = File(...)):
    # Read the tiny face crop from React
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    # Get embedding for the live face
    faces = face_app.get(img)
    if not faces:
        return {"status": "no_face"}
    
    live_vec = faces[0].normed_embedding.astype('float32').reshape(1, -1)
    
    # Search the RAM index
    D, I = index.search(live_vec, k=1)
    
    if D[0][0] > 0.65: # Threshold for match
        return {"status": "success", "htNo": index_keys[I[0][0]], "score": float(D[0][0])}
    
    return {"status": "unknown"}