import os
import cv2
import numpy as np
import requests
import base64
import gc
from fastapi import FastAPI
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

# 1. Use the absolute smallest detection model (scrfd_500m)
# 2. Use 'mobilenet' for recognition - it's a fraction of the size of 'buffalo'
face_app = FaceAnalysis(det_name='scrfd_500m', rec_name='mobilenet', providers=['CPUExecutionProvider'])

# 3. Use a tiny 320x320 detection window to drastically cut RAM usage
face_app.prepare(ctx_id=-1, det_size=(320, 320))

APPSCRIPT_URL = os.getenv("APPSCRIPT_URL")
student_db = []

@app.on_event("startup")
def sync_data():
    global student_db
    try:
        response = requests.get(f"{APPSCRIPT_URL}?action=getStudents")
        student_db = response.json()
        gc.collect() # Force garbage collection immediately after sync
    except:
        pass

class FrameData(BaseModel):
    image: str

@app.post("/verify")
async def verify_face(data: FrameData):
    try:
        # Decode only what is necessary
        encoded_data = data.image.split(',')[1]
        nparr = np.frombuffer(base64.b64decode(encoded_data), np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        faces = face_app.get(img)
        if not faces:
            return {"match": False}

        input_embedding = faces[0].normed_embedding

        for student in student_db:
            if not student.get("face_vector"): continue
            sim = np.dot(input_embedding, np.array(student["face_vector"]))
            
            if sim > 0.42:
                requests.post(APPSCRIPT_URL, json={"action": "markAttendance", "rollNo": student["rollNo"]})
                return {"match": True, "name": student["name"]}

        return {"match": False}
    finally:
        # Critical: Clear memory after every single request
        gc.collect()

@app.get("/")
def health():
    return {"status": "Live", "mode": "low_memory"}
