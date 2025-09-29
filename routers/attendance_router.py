import os
import tempfile
from fastapi import APIRouter, UploadFile, Depends, HTTPException
from sqlalchemy.orm import Session
from services.student_service import extract_frames_from_video
from services.face_service import recognize_students_from_frames
from services.session_service import session_manager
from database import get_db

router = APIRouter(prefix="/attendance", tags=["Attendance"])

@router.post("/upload-video")
async def upload_video(file: UploadFile):
    """
    Upload a video file and create a streaming session.
    
    Returns:
        session_id: Use this ID to connect to WebSocket for streaming and recognition
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail="Filename is required")
    
    # Validate file type
    allowed_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm'}
    file_extension = os.path.splitext(file.filename)[1].lower()
    
    if file_extension not in allowed_extensions:
        raise HTTPException(
            status_code=400, 
            detail=f"Unsupported file type. Allowed: {', '.join(allowed_extensions)}"
        )
    
    # Read file content
    file_content = await file.read()
    
    if len(file_content) == 0:
        raise HTTPException(status_code=400, detail="Empty file uploaded")
    
    # Create session for WebSocket streaming
    session_id = session_manager.create_session(file.filename, file_content)
    
    return {
        "message": "Video uploaded successfully",
        "session_id": session_id,
        "filename": file.filename,
        "file_size": len(file_content),
        "websocket_url": f"/ws/stream-video/{session_id}"
    }

@router.get("/session/{session_id}")
async def get_session_status(session_id: str):
    """Get the status of a video streaming session."""
    session_data = session_manager.get_session(session_id)
    
    if not session_data:
        raise HTTPException(status_code=404, detail="Session not found")
    
    # Return session info without sensitive file paths
    return {
        "session_id": session_id,
        "filename": session_data["filename"],
        "status": session_data["status"],
        "progress": session_data["progress"],
        "total_frames": session_data["total_frames"],
        "recognized_students": session_data["recognized_students"],
        "created_at": session_data["created_at"].isoformat()
    }

@router.post("/process-video-direct")
async def process_video_direct(file: UploadFile, db: Session = Depends(get_db)):
    """
    LEGACY: Direct video processing without streaming (for backward compatibility).
    Use upload-video + WebSocket streaming for better user experience.
    """
    # Save uploaded file temporarily
    temp_dir = tempfile.mkdtemp()
    video_path = os.path.join(temp_dir, file.filename)

    with open(video_path, "wb") as f:
        content = await file.read()
        f.write(content)

    try:
        # Extract frames
        frames = extract_frames_from_video(video_path, frame_skip=10)

        # Recognize students from frames
        recognized_students = recognize_students_from_frames(frames, db)

        return {"recognized_students": recognized_students}
    
    finally:
        # Clean up temp file
        if os.path.exists(video_path):
            os.remove(video_path)
        if os.path.exists(temp_dir):
            os.rmdir(temp_dir)
