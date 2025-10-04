# services/video_service.py
import os
import shutil
from fastapi import UploadFile

UPLOAD_DIR = os.path.join(os.path.dirname(__file__), "..", "temp_videos")
UPLOAD_DIR = os.path.abspath(UPLOAD_DIR)

def save_uploaded_video(file: UploadFile) -> str:
    """
    Save uploaded video to temp_videos/ and return full path.
    """
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    save_path = os.path.join(UPLOAD_DIR, file.filename)

    with open(save_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    return save_path

def get_video_full_path(filename: str) -> str:
    """
    Return full path for a filename in temp_videos.
    """
    return os.path.join(UPLOAD_DIR, filename)


