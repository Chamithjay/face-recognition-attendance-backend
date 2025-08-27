import os
import tempfile
from fastapi import APIRouter, UploadFile, Depends
from sqlalchemy.orm import Session
from services.student_service import extract_frames_from_video
from services.face_service import recognize_students_from_frames
from database import get_db

router = APIRouter(prefix="/attendance", tags=["Attendance"])

@router.post("/upload-video")
async def upload_video(file: UploadFile, db: Session = Depends(get_db)):
    # Save uploaded file temporarily
    temp_dir = tempfile.mkdtemp()
    video_path = os.path.join(temp_dir, file.filename)

    with open(video_path, "wb") as f:
        f.write(file.file.read())

    # Extract frames
    frames = extract_frames_from_video(video_path, frame_skip=10)

    # Recognize students from frames
    recognized_students = recognize_students_from_frames(frames, db)

    return {"recognized_students": recognized_students}
