"""
Video file handling utilities for upload and storage management.
"""

import os
import shutil
from fastapi import UploadFile

UPLOAD_DIR = os.path.join(os.path.dirname(__file__), "..", "temp_videos")
UPLOAD_DIR = os.path.abspath(UPLOAD_DIR)


def save_uploaded_video(file: UploadFile) -> str:
    """
    Save uploaded video file to temporary storage directory.
    
    Args:
        file: Uploaded video file from FastAPI
        
    Returns:
        Absolute path to saved video file
    """
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    save_path = os.path.join(UPLOAD_DIR, file.filename)

    with open(save_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    return save_path

def get_video_full_path(filename: str) -> str:
    """
    Construct absolute path for video file in temporary storage.
    
    Args:
        filename: Name of video file
        
    Returns:
        Absolute path to video file
    """
    return os.path.join(UPLOAD_DIR, filename)


