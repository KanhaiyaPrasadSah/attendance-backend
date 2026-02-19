import os
import cv2
import numpy as np
import requests
import base64
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from insightface.app import FaceAnalysis

# 1. Initialize FastAPI
app = FastAPI()

# Enable CORS for React Frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# 2. Configuration & Environment Variables
# Use buffalo_s (small) to avoid Out of Memory errors
face_app = FaceAnalysis(name='buffalo_s', providers=['CPUExecutionProvider'])
face_app.prepare(ctx_id=-1)

APPSCRIPT_URL = os.getenv("APPSCRIPT_URL")
student_db = []  # Local cache of student data

# 3. Data Models
class FrameData(BaseModel):
    image: str  # Base64 encoded image from webcam

# 4. Helper: Sync Data from Google Sheets
@app.on_event("startup")
def sync_data():
    global student_db
    try:
        response = requests.get(f"{APPSCRIPT_URL}?action=getStudents")
        data = response.json()
        student_db = data
        print(f"✅ Sync Complete: {len(student_db)} students loaded.")
    except Exception as e:
        print(f"❌ Sync Failed: {e}")

# 5. Core Logic: Recognition Endpoint
@app.post("/verify")
async def verify_face(data: FrameData):
    try:
        # Decode base64 image
        encoded_data = data.image.split(',')[1]
        nparr = np.frombuffer(base64.b64decode(encoded_data), np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        # Detect faces
        faces = face_app.get(img)
        if not faces:
            return {"match": False, "message": "No face detected"}

        input_embedding = faces[0].normed_embedding

        # Compare with Database
        for student in student_db:
            if not student.get("face_vector"): continue
            
            db_vector = np.array(student["face_vector"])
            # Calculate Cosine Similarity
            sim = np.dot(input_embedding, db_vector)
            
            if sim > 0.45:  # Similarity Threshold
                # Send Attendance to Google Sheets
                requests.post(APPSCRIPT_URL, json={
                    "action": "markAttendance",
                    "rollNo": student["rollNo"]
                })
                return {
                    "match": True, 
                    "name": student["name"], 
                    "rollNo": student["rollNo"]
                }

        return {"match": False, "message": "Face not recognized"}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
def health():
    return {"status": "Live", "model": "buffalo_s", "students_loaded": len(student_db)}
