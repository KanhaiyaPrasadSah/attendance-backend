import numpy as np
import faiss
import requests
import cv2
import os
from fastapi import FastAPI, UploadFile, File
from insightface.app import FaceAnalysis
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# Enable CORS so your React Frontend can talk to this Backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# 1. Initialize AI Model (Buffalo_L)
# This model converts a face image into a 512-dimension mathematical vector
face_app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
face_app.prepare(ctx_id=-1, det_size=(640, 640))

# 2. In-Memory Search Index (FAISS)
index = faiss.IndexFlatIP(512) 
index_keys = [] 

# 3. Get your AppScript URL from Render Environment Variables
APPSCRIPT_URL = os.getenv("APPSCRIPT_URL")

@app.on_event("startup")
async def sync_database():
    """Syncs Column T from Google Sheets into Server RAM on startup"""
    global index_keys
    print("🚀 Starting RAM Sync with Google Sheets...")
    try:
        response = requests.get(f"{APPSCRIPT_URL}?action=getVectors").json()
        if response.get("status") == "success":
            embeddings = []
            for item in response["data"]:
                if item["vector"]:
                    # Convert string vector from Col T to Numpy Array
                    vec = np.fromstring(item["vector"], sep=',').astype('float32')
                    embeddings.append(vec)
                    index_keys.append(item["htNo"])
            
            if embeddings:
                index.add(np.stack(embeddings))
                print(f"✅ Sync Complete: {len(index_keys)} students loaded in RAM.")
            else:
                print("⚠️ Sync Warning: No face vectors found in Sheet.")
    except Exception as e:
        print(f"❌ RAM Sync Failed: {e}")

@app.post("/identify")
async def identify(file: UploadFile = File(...)):
    """Receives face crop from React and finds the student ID"""
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    faces = face_app.get(img)
    if not faces:
        return {"status": "no_face"}
    
    # Generate vector for the live face
    live_vec = faces[0].normed_embedding.astype('float32').reshape(1, -1)
    
    # Search RAM index for the closest match
    D, I = index.search(live_vec, k=1)
    
    # 0.65 is the sweet spot for Buffalo_L accuracy
    if D[0][0] > 0.65:
        return {
            "status": "success", 
            "htNo": index_keys[I[0][0]], 
            "confidence": float(D[0][0])
        }
    
    return {"status": "unknown"}

@app.get("/")
def health_check():
    return {"status": "online", "students_loaded": len(index_keys)}