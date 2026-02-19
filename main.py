import os
import cv2
import numpy as np
import requests
import base64
import gc
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from insightface.app import FaceAnalysis

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Use 'antelopev2' - it is extremely small and memory-efficient
# This is critical for staying under 512MB
face_app = FaceAnalysis(name='antelopev2', providers=['CPUExecutionProvider'])
face_app.prepare(ctx_id=-1, det_size=(320, 320)) # Smaller detection size saves RAM

APPSCRIPT_URL = os.getenv("APPSCRIPT_URL")
student_db = []

@app.on_event("startup")
def sync_data():
    global student_db
    try:
        response = requests.get(f"{APPSCRIPT_URL}?action=getStudents")
        student_db = response.json()
        print(f"✅ Sync Complete: {len(student_db)} students loaded.")
        gc.collect() # Force clear memory after sync
    except Exception as e:
        print(f"❌ Sync Failed: {e}")

class FrameData(BaseModel):
    image: str

@app.post("/verify")
async def verify_face(data: FrameData):
    try:
        encoded_data = data.image.split(',')[1]
        nparr = np.frombuffer(base64.b64decode(encoded_data), np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        faces = face_app.get(img)
        if not faces:
            return {"match": False, "message": "No face detected"}

        input_embedding = faces[0].normed_embedding

        for student in student_db:
            if not student.get("face_vector"): continue
            db_vector = np.array(student["face_vector"])
            sim = np.dot(input_embedding, db_vector)
            
            if sim > 0.4:
                requests.post(APPSCRIPT_URL, json={
                    "action": "markAttendance",
                    "rollNo": student["rollNo"]
                })
                return {"match": True, "name": student["name"], "rollNo": student["rollNo"]}

        return {"match": False, "message": "Not recognized"}
    finally:
        gc.collect() # Clean up after every request to stay under 512MB

@app.get("/")
def health():
    return {"status": "Live", "model": "antelopev2"}
