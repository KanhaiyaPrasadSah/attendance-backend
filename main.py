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

# Define face_app as None initially to speed up server start
face_app = None

def get_face_app():
    global face_app
    if face_app is None:
        # Initializing only the absolute necessary components
        face_app = FaceAnalysis(
            det_name='scrfd_500m', 
            rec_name='mobilenet', 
            providers=['CPUExecutionProvider']
        )
        face_app.prepare(ctx_id=-1, det_size=(160, 160))
    return face_app

APPSCRIPT_URL = os.getenv("APPSCRIPT_URL")
student_db = []

@app.on_event("startup")
async def startup_event():
    # Load students but don't load the AI model yet to avoid port-binding timeout
    global student_db
    try:
        response = requests.get(f"{APPSCRIPT_URL}?action=getStudents", timeout=10)
        student_db = response.json()
        gc.collect()
    except Exception as e:
        print(f"Startup warning: {e}")

class FrameData(BaseModel):
    image: str

@app.post("/verify")
async def verify_face(data: FrameData):
    global student_db
    try:
        # Lazy load the AI model here
        handler = get_face_app()
        
        encoded_data = data.image.split(',')[1]
        nparr = np.frombuffer(base64.b64decode(encoded_data), np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        faces = handler.get(img)
        if not faces:
            return {"match": False, "message": "No face detected"}

        input_embedding = faces[0].normed_embedding

        for student in student_db:
            if not student.get("face_vector"): continue
            sim = np.dot(input_embedding, np.array(student["face_vector"]))
            
            if sim > 0.40:
                requests.post(APPSCRIPT_URL, json={"action": "markAttendance", "rollNo": student["rollNo"]})
                return {"match": True, "name": student["name"]}

        return {"match": False}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        gc.collect()

@app.get("/")
def health():
    return {"status": "Live", "port_detected": True}
